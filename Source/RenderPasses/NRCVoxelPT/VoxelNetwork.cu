#ifndef __NVCC__        // defined by nvcc complier 
#define __NVCC__
#endif

#include "VoxelNetwork.h"
#include "Helpers.h"
#include "Parameters.h"
#include "json/json.hpp"

#include <curand.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>

using namespace tcnn;
using precision_t = tcnn::network_precision_t;

#define GPUMatrix GPUMatrix<float, CM>

namespace {
    // cuda related
    cudaStream_t inference_stream;
    cudaStream_t training_stream;
    cudaStream_t *voxel_inference_stream;
    cudaStream_t *voxel_training_stream;

    curandGenerator_t rng;

    struct _Network {
        std::vector<std::shared_ptr<NetworkWithInputEncoding<precision_t>>> voxel_network;
        std::vector<std::shared_ptr<Loss<precision_t>>> voxel_loss;
        std::vector <std::shared_ptr<Optimizer<precision_t>>> voxel_optimizer;
        std::vector <std::shared_ptr<Trainer<float, precision_t, precision_t>>> voxel_trainer;
    };

    struct _Memory {
        // the GPUMatrix class supports MxN matrices only
        // the GPUMatrix store in a continuous area in memory, either row major or column major
        GPUMatrix* training_data = nullptr;
        GPUMatrix* training_target = nullptr;
        GPUMatrix* inference_data = nullptr;
        GPUMatrix* inference_target = nullptr;
        GPUMatrix* training_self_query = nullptr;
        GPUMatrix* training_self_pred = nullptr;
        GPUMemory<float>* random_seq = nullptr;
    };

    struct _Counter {   // pinned memory on device
        uint32_t training_query_count;
        uint32_t training_sample_count;
        uint32_t inference_query_count;

        uint32_t* training_query_counter;
        uint32_t* training_sample_counter;
        uint32_t* inference_query_counter;
    };

    _Memory* mMemory;
    _Network* mNetwork;
    _Counter* mCounter;     // pinned memory via cudaHostAlloc
    
}

template <typename T>
__device__ void copyQuery(T* data, const NRC::RadianceQuery* query) {
    // use naive copy kernel since memcpy has bad performance on small datas.

    data[0] = query->pos.x, data[1] = query->pos.y, data[2] = query->pos.z;
    data[3] = query->dir.x, data[4] = query->dir.y;
#if AUX_INPUTS
    data[5] = query->roughness;
    data[6] = query->normal.x, data[7] = query->normal.y;
    data[8] = query->diffuse.x, data[9] = query->diffuse.y, data[10] = query->diffuse.z;
    data[11] = query->specular.x, data[12] = query->specular.y, data[13] = query->specular.z;
#endif
}

// linear kernels with only x-dim not 1. must be called using linear_kernal()
// blockDim = 128, threadIdx is the index of a thread within a thread block, i.e. in [0, 128)
// reference linear_kernel() for details.
// stride: input dim
template <uint32_t stride, typename T = float>
__global__ void generateBatchSequential(uint32_t n_elements, uint32_t offset,
    NRC::RadianceQuery* queries, T* data) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset < n_elements) {
        uint32_t data_index = i * stride, query_index = i + offset;
        copyQuery(&data[data_index], &queries[query_index]);
    }
}

template <uint32_t stride, typename T = float>
__global__ void generateTrainingDataFromSamples(uint32_t n_elements, uint32_t offset,
    NRC::RadianceSample* samples, NRC::RadianceQuery* self_queries, T* self_query_pred,
    T* training_data, T* training_target, uint32_t* training_sample_counter, uint32_t* self_query_counter,
    float* random_indices = nullptr) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset > n_elements) return;
    int data_index = i * stride, sample_index = i + offset;
    if (random_indices)
        sample_index = (1 - random_indices[sample_index]) * *training_sample_counter;

    int pred_index = samples[sample_index].idx; // pred_index == -1 if a self-query is not needed.

    if (sample_index < *training_sample_counter) {
        float3 factor = samples[sample_index].a, bias = samples[sample_index].b;
        uint32_t output_index = i * 3;

        copyQuery(&training_data[data_index], &samples[sample_index].query);

        float3 pred_radiance = { 0, 0, 0 };
        if (pred_index >= 0)    // else the sample doesn't contain a self query.
            pred_radiance = { self_query_pred[pred_index * 3], self_query_pred[pred_index * 3 + 1], self_query_pred[pred_index * 3 + 2] };
#if REFLECTANCE_FACT
        float3 reflectance = samples[sample_index].query.diffuse + samples[sample_index].query.specular;
        if (pred_index >= 0)
            // restore self-query from reflectance factorization...
            pred_radiance = pred_radiance * (self_queries[pred_index].diffuse + self_queries[pred_index].specular);
        float3 radiance = safe_div(pred_radiance * factor + bias, reflectance);
#else
        float3 radiance = pred_radiance * factor + bias;
#endif
        * (float3*)&training_target[output_index] = radiance;
    }
}

template <typename T>
__global__ void mapIndexedRadianceToScreen(uint32_t n_elements, T* data,
    cudaSurfaceObject_t output, Falcor::uint2* pixels) {
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i > n_elements) return;
    uint32_t px = pixels[i].x, py = pixels[i].y;
    uint32_t data_index = i * 3;
    float4 radiance = { data[data_index], data[data_index + 1], data[data_index + 2], 1.0f };
    surf2Dwrite(radiance, output, (int)sizeof(float4) * px, py);

}

using namespace NRC::Parameters;

namespace NRC {
    VoxelNetwork::VoxelNetwork(json config)
    {
        CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
        CUDA_CHECK_THROW(cudaStreamCreate(&training_stream));
        //training_stream = inference_stream;

        curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(rng, random_seed);
        curandSetStream(rng, training_stream);

        initializeNetwork(config);
    }

    VoxelNetwork::~VoxelNetwork()
    {
        delete mNetwork;
        delete mMemory;
    }

    void VoxelNetwork::registerResource(NRCResource resource) {
        mResource = resource;
    }

    void VoxelNetwork::initializeNetwork(json net_config)
    {
        mNetwork = new _Network();
        mMemory = new _Memory();

        mNetwork->voxel_network.resize(voxel_param.voxel_num);
        mNetwork->voxel_loss.resize(voxel_param.voxel_num);
        mNetwork->voxel_optimizer.resize(voxel_param.voxel_num);
        mNetwork->voxel_trainer.resize(voxel_param.voxel_num);

        cudaHostAlloc((void**)&mCounter, sizeof(_Counter), cudaHostAllocDefault);
        cudaHostAlloc(&mCounter->inference_query_counter, voxel_param.voxel_num * sizeof(uint32_t), cudaHostAllocDefault);
        cudaHostAlloc(&mCounter->training_query_counter, voxel_param.voxel_num * sizeof(uint32_t), cudaHostAllocDefault);
        cudaHostAlloc(&mCounter->training_sample_counter, voxel_param.voxel_num * sizeof(uint32_t), cudaHostAllocDefault);

        // network parameters
        json loss_opts = net_config.value("loss", json::object());
        json optimizer_opts = net_config.value("optimizer", json::object());
        json network_opts = net_config.value("network", json::object());
        json encoding_opts = net_config.value("encoding", json::object());
        m_learning_rate = optimizer_opts["nested"].value("learning_rate", 1e-3);

        std::cout << "VoxelNetwork::Initialize networks for " + std::to_string(voxel_param.voxel_num) + " voxels!" << std::endl;

        for (int i = 0; i < voxel_param.voxel_num; i++) {
            //auto [loss, optimizer, network, trainer] = create_from_config(input_dim, output_dim, net_config);
            mNetwork->voxel_loss[i] = std::shared_ptr<Loss<precision_t>>(create_loss<precision_t>(loss_opts));
            mNetwork->voxel_optimizer[i] = std::shared_ptr<Optimizer<precision_t>>(create_optimizer<precision_t>(optimizer_opts));
            mNetwork->voxel_network[i] = std::make_shared<NetworkWithInputEncoding<precision_t>>(input_dim, output_dim, encoding_opts, network_opts);
            mNetwork->voxel_trainer[i] = std::make_shared<Trainer<float, precision_t, precision_t>>(
                mNetwork->voxel_network[i], mNetwork->voxel_optimizer[i], mNetwork->voxel_loss[i]);
        }

        mMemory->training_data = new GPUMatrix(input_dim, batch_size);
        mMemory->training_target = new GPUMatrix(output_dim, batch_size);
        mMemory->inference_data = new GPUMatrix(input_dim, resolution);
        mMemory->inference_target = new GPUMatrix(output_dim, resolution);
        mMemory->training_self_query = new GPUMatrix(input_dim, self_query_batch_size);
        mMemory->training_self_pred = new GPUMatrix(output_dim, self_query_batch_size);

        mMemory->random_seq = new GPUMemory<float>(n_train_batch * batch_size);
        curandGenerateUniform(rng, mMemory->random_seq->data(), n_train_batch * batch_size);
    }

    void VoxelNetwork::reset()
    {
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        for (int i = 0; i < voxel_param.voxel_num; i++)
            mNetwork->voxel_trainer[i]->initialize_params();
    }

    void VoxelNetwork::prepare()
    {
        cudaMemcpy(mCounter, mResource.counterBufferPtr, sizeof(uint32_t) * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(mCounter->inference_query_counter, mResource.inferenceQueryCounter, sizeof(uint32_t) * voxel_param.voxel_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(mCounter->training_sample_counter, mResource.trainingSampleCounter, sizeof(uint32_t) * voxel_param.voxel_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(mCounter->training_query_counter, mResource.trainingQueryCounter, sizeof(uint32_t) * voxel_param.voxel_num, cudaMemcpyDeviceToHost);
    }

    void VoxelNetwork::inference()
    {
        uint32_t n_elements = mCounter->inference_query_count;
        if (!n_elements) return;

        CUDA_CHECK_THROW(cudaStreamSynchronize(inference_stream));
    }

    void VoxelNetwork::train(float& loss)
    {
        CUDA_CHECK_THROW(cudaStreamSynchronize(training_stream));
    }

    void VoxelNetwork::debug()
    {
        printf("Total inference queries: %d, total training samples: %d\n", mCounter->inference_query_count, mCounter->training_sample_count);
        for (int i = 0; i < voxel_param.voxel_num; i++) {
            printf("current voxel: #%d, inference queries: %d, training samples: %d\n",
                i, mCounter->inference_query_counter[i], mCounter->training_sample_counter[i]);
        }
    }
}
