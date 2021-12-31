#ifndef __NVCC__        // defined by nvcc complier 
#define __NVCC__
#endif

#include "VoxelNetwork.h"
#include "Helpers.h"
#include "Parameters.h"

#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <json/json.hpp>
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
        std::vector<std::shared_ptr<Optimizer<precision_t>>> voxel_optimizer;
        std::vector<std::shared_ptr<Trainer<float, precision_t, precision_t>>> voxel_trainer;
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

    struct _Stat {
        uint32_t inference_query_count;
        uint32_t training_sample_count;
        uint32_t training_query_count;

        thrust::host_vector<uint32_t> inference_query_counter_h;
        thrust::host_vector<uint32_t> training_sample_counter_h;
        thrust::host_vector<uint32_t> training_query_counter_h;

        thrust::device_vector<uint32_t> inference_query_counter;
        thrust::device_vector<uint32_t> training_sample_counter;
        thrust::device_vector<uint32_t> training_query_counter;
    };

    _Stat mStat;
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

template <uint32_t stride, typename T = float>
__global__ void sortForInference(const uint32_t n_elements, const NRC::RadianceQuery* queries, const uint32_t* offsets,
    const uint32_t* counters, float* output, const uint32_t n_voxels) {

}

namespace NRC {
    using namespace Parameters;

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

        mStat.inference_query_counter_h.resize(voxel_param.voxel_num, 0);
        mStat.training_sample_counter_h.resize(voxel_param.voxel_num, 0);
        mStat.training_query_counter_h.resize(voxel_param.voxel_num, 0);
        mStat.inference_query_counter.resize(voxel_param.voxel_num, 0);
        mStat.training_sample_counter.resize(voxel_param.voxel_num, 0);
        mStat.training_query_counter.resize(voxel_param.voxel_num, 0);

        mNetwork->voxel_network.resize(voxel_param.voxel_num);
        mNetwork->voxel_loss.resize(voxel_param.voxel_num);
        mNetwork->voxel_optimizer.resize(voxel_param.voxel_num);
        mNetwork->voxel_trainer.resize(voxel_param.voxel_num);

        cudaHostAlloc((void**)&mCounter, sizeof(_Counter), cudaHostAllocDefault);
        //cudaHostAlloc(&mCounter->inference_query_counter, voxel_param.voxel_num * sizeof(uint32_t), cudaHostAllocDefault);
        //cudaHostAlloc(&mCounter->training_query_counter, voxel_param.voxel_num * sizeof(uint32_t), cudaHostAllocDefault);
        //cudaHostAlloc(&mCounter->training_sample_counter, voxel_param.voxel_num * sizeof(uint32_t), cudaHostAllocDefault);

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
        // currently cost <<1ms, seems good φ(゜▽゜*)♪
        //cudaMemcpy(mCounter, mResource.counterBufferPtr, sizeof(uint32_t) * 3, cudaMemcpyDeviceToHost);
        //cudaMemcpy(mCounter->inference_query_counter, mResource.inferenceQueryCounter, sizeof(uint32_t) * voxel_param.voxel_num, cudaMemcpyDeviceToHost);
        //cudaMemcpy(mCounter->training_sample_counter, mResource.trainingSampleCounter, sizeof(uint32_t) * voxel_param.voxel_num, cudaMemcpyDeviceToHost);
        //cudaMemcpy(mCounter->training_query_counter, mResource.trainingQueryCounter, sizeof(uint32_t) * voxel_param.voxel_num, cudaMemcpyDeviceToHost);

        // thrust's copy routine costs similar to 1 cudaMemcpy. 
        thrust::copy(mResource.inferenceQueryCounter, mResource.inferenceQueryCounter + voxel_param.voxel_num, mStat.inference_query_counter.begin());
        thrust::copy(mResource.trainingSampleCounter, mResource.trainingSampleCounter + voxel_param.voxel_num, mStat.training_sample_counter.begin());
        thrust::copy(mResource.trainingQueryCounter, mResource.trainingQueryCounter + voxel_param.voxel_num, mStat.training_query_counter.begin());

        thrust::copy(mStat.inference_query_counter.begin(), mStat.inference_query_counter.end(), mStat.inference_query_counter_h.begin());
        thrust::copy(mStat.training_sample_counter.begin(), mStat.training_sample_counter.end(), mStat.training_sample_counter_h.begin());
        thrust::copy(mStat.training_query_counter.begin(), mStat.training_query_counter.end(), mStat.training_query_counter_h.begin());

        mStat.inference_query_count = thrust::reduce(mStat.inference_query_counter_h.begin(), mStat.inference_query_counter_h.end(), 0, thrust::plus<uint32_t>());
        mStat.training_query_count = thrust::reduce(mStat.training_query_counter_h.begin(), mStat.training_query_counter_h.end(), 0, thrust::plus<uint32_t>());
        mStat.training_sample_count = thrust::reduce(mStat.training_sample_counter_h.begin(), mStat.training_sample_counter_h.end(), 0, thrust::plus<uint32_t>());
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
        printf("Total inference queries: %d, total training samples: %d\n", mStat.inference_query_count, mStat.training_sample_count);
        for (int i = 0; i < voxel_param.voxel_num; i++) {
            printf("thrust@current voxel: #%d, inference queries: %d, training samples: %d\n",
                i, mStat.inference_query_counter_h[i], mStat.training_sample_counter_h[i]);
            // the thrust counter on device seemes not print correctly, which may be not an issue.
        }
    }
}
