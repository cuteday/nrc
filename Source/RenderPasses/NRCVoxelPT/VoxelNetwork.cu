#ifndef __NVCC__        // defined by nvcc complier 
#define __NVCC__
#endif

#include "VoxelNetwork.h"
#include "Helpers.h"
#include "Parameters.h"

#include <curand.h>

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <json/json.hpp>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>

using namespace tcnn;
using precision_t = tcnn::network_precision_t;
using thrust::device_vector;
using thrust::host_vector;

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

        // thrust's place
        device_vector<NRC::RadianceSample*> sample_storage;
        device_vector<uint64_t> sample_voxel_counter;
        host_vector<uint64_t> sample_voxel_counter_h;
    };

    struct _Stat {
        uint32_t inference_query_count;
        uint32_t training_sample_count;
        uint32_t training_query_count;

        device_vector<NRC::RadianceQuery> inference_query;
        device_vector<uint32_t> inference_sorting_counter;

        device_vector<uint32_t> inference_query_voxel;
        device_vector<uint32_t> training_sample_voxel;
        device_vector<uint32_t> training_query_voxel;

        host_vector<uint32_t> inference_query_counter_h;
        host_vector<uint32_t> training_sample_counter_h;
        host_vector<uint32_t> training_query_counter_h;

        host_vector<uint32_t> inference_query_offset;
        host_vector<uint32_t> training_sample_offset;
        host_vector<uint32_t> training_query_offset;

        device_vector<uint32_t> inference_query_counter;
        device_vector<uint32_t> training_sample_counter;
        device_vector<uint32_t> training_query_counter;
    };

    _Stat mStat;
    _Memory* mMemory;
    _Network* mNetwork;
    //_Counter* mCounter;     // pinned memory via cudaHostAlloc 
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

struct get_voxel_index {    // thrust custom transform functor
     __host__ __device__ uint32_t operator() (NRC::RadianceQuery query) { return query.voxel_idx; }
};

template <uint32_t stride, typename T = float>
__global__ void sortForInference(const uint32_t n_elements, const NRC::RadianceQuery* queries, const device_vector<uint32_t> &offsets,
    uint32_t* counters, float* output, const uint32_t n_voxels) {
    
}

template <uint32_t stride>
__global__ void generateInferenceData(const uint32_t n_elements, const NRC::RadianceQuery* queries,
    float* output) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n_elements) return;
    copyQuery(&output[i * stride], &queries[i]);
}

template <typename T>
__global__ void mapRadianceToScreen(const uint32_t n_elements, const NRC::RadianceQuery* queries,
    const float* data, cudaSurfaceObject_t output) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n_elements) return;
    uint32_t px = queries[i].pixel.x, py = queries[i].pixel.y;
    uint32_t data_index = i * 3;
    float4 radiance = { data[data_index], data[data_index + 1], data[data_index + 2], 1.0f };
    surf2Dwrite(radiance, output, (int)sizeof(float4) * px, py);
}

template <typename T = float>
__global__ void sortForTraining(const uint32_t n_elements, const NRC::RadianceSample* samples,
    uint64_t* counters, NRC::RadianceSample* sample_storage[],
    const uint32_t n_voxels, const uint32_t n_max_samples) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n_elements) return;
    uint32_t voxel_id = samples[i].query.voxel_idx;
    uint32_t sample_id = atomicAdd(counters + voxel_id, 1);
    uint32_t write_id = sample_id % n_max_samples;
    sample_storage[voxel_id][write_id] = samples[i];
}

template <uint32_t stride, typename T = float>
__global__ void generateTrainingData(const uint32_t n_elements, const NRC::RadianceSample* samples,
    uint32_t n_samples, float* input_data, float* output_data, float* random_sequence = nullptr) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n_elements) return;

    //const NRC::RadianceSample* samples = *samples;

    uint32_t sample_id = random_sequence ? (1. - random_sequence[i]) * n_samples : (float)i / n_elements * n_samples;
    int input_data_index = i * stride, output_data_index = i * 3;
    // copy input data
    copyQuery(&input_data[input_data_index], &samples[sample_id].query);
    // copy output data
    float3 factor = samples[sample_id].a, bias = samples[sample_id].b;
    float3 reflectance = samples[sample_id].query.diffuse + samples[sample_id].query.specular;
    float3 radiance = bias;
#if REFLECTANCE_FACT
    radiance = safe_div(radiance, reflectance);
#endif
    *(float3*)&output_data[output_data_index] = radiance;
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

        mStat.inference_query.resize(max_inference_query_size);

        uint32_t n_voxels = voxel_param.voxel_num;

        mStat.inference_query_counter_h.resize(n_voxels, 0);
        mStat.training_sample_counter_h.resize(n_voxels, 0);
        mStat.training_query_counter_h.resize(n_voxels, 0);
        mStat.inference_query_counter.resize(n_voxels, 0);
        mStat.training_sample_counter.resize(n_voxels, 0);
        mStat.training_query_counter.resize(n_voxels, 0);

        mStat.inference_query_offset.resize(n_voxels);
        mStat.inference_query_voxel.reserve(max_inference_query_size);

        mMemory->sample_storage.reserve(n_voxels);
        for (int i = 0; i < n_voxels; i++) {
            RadianceSample* samples;
            cudaMalloc(&samples, sizeof(RadianceSample) * max_training_sample_voxel);
            mMemory->sample_storage.push_back(samples);
        }
        mMemory->sample_voxel_counter.resize(n_voxels);
        mMemory->sample_voxel_counter_h.resize(n_voxels);

        mNetwork->voxel_network.resize(n_voxels);
        mNetwork->voxel_loss.resize(n_voxels);
        mNetwork->voxel_optimizer.resize(n_voxels);
        mNetwork->voxel_trainer.resize(n_voxels);

        // network parameters
        json loss_opts = net_config.value("loss", json::object());
        json optimizer_opts = net_config.value("optimizer", json::object());
        json network_opts = net_config.value("network", json::object());
        json encoding_opts = net_config.value("encoding", json::object());
        m_learning_rate = optimizer_opts["nested"].value("learning_rate", 1e-3);

        std::cout << "VoxelNetwork::Initialize networks for " + std::to_string(n_voxels) + " voxels!" << std::endl;

        for (int i = 0; i < n_voxels; i++) {
            //auto [loss, optimizer, network, trainer] = create_from_config(input_dim, output_dim, net_config);
            mNetwork->voxel_loss[i] = std::shared_ptr<Loss<precision_t>>(create_loss<precision_t>(loss_opts));
            mNetwork->voxel_optimizer[i] = std::shared_ptr<Optimizer<precision_t>>(create_optimizer<precision_t>(optimizer_opts));
            mNetwork->voxel_network[i] = std::make_shared<NetworkWithInputEncoding<precision_t>>(input_dim, output_dim, encoding_opts, network_opts);
            mNetwork->voxel_trainer[i] = std::make_shared<Trainer<float, precision_t, precision_t>>(
                mNetwork->voxel_network[i], mNetwork->voxel_optimizer[i], mNetwork->voxel_loss[i]);
        }

        mMemory->training_data = new GPUMatrix(input_dim, batch_size);
        mMemory->training_target = new GPUMatrix(output_dim, batch_size);
        mMemory->inference_data = new GPUMatrix(input_dim, max_inference_query_size);
        mMemory->inference_target = new GPUMatrix(output_dim, max_inference_query_size);
        mMemory->training_self_query = new GPUMatrix(input_dim, self_query_batch_size);
        mMemory->training_self_pred = new GPUMatrix(output_dim, self_query_batch_size);
        mMemory->random_seq = new GPUMemory<float>(max_training_sample_voxel);
    }

    void VoxelNetwork::reset()
    {
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        for (int i = 0; i < voxel_param.voxel_num; i++)
            mNetwork->voxel_trainer[i]->initialize_params();
        
    }

    void VoxelNetwork::prepare()
    {
        // currently cost << 1ms, seems good φ(゜▽゜*)♪

        //printf("VoxelNetwork::preparing nrc resources\n");
        //printf("VoxelNetwork::copying nrc resource counters\n");
        // thrust's copy routine costs similar to 1 cudaMemcpy. 
        thrust::copy(mResource.inferenceQueryCounter, mResource.inferenceQueryCounter + voxel_param.voxel_num, mStat.inference_query_counter.begin());
        thrust::copy(mResource.trainingSampleCounter, mResource.trainingSampleCounter + voxel_param.voxel_num, mStat.training_sample_counter.begin());
        thrust::copy(mResource.trainingQueryCounter, mResource.trainingQueryCounter + voxel_param.voxel_num, mStat.training_query_counter.begin());
        //printf("VoxelNetwork::transfer counters to device\n");
        thrust::copy(mStat.inference_query_counter.begin(), mStat.inference_query_counter.end(), mStat.inference_query_counter_h.begin());
        thrust::copy(mStat.training_sample_counter.begin(), mStat.training_sample_counter.end(), mStat.training_sample_counter_h.begin());
        thrust::copy(mStat.training_query_counter.begin(), mStat.training_query_counter.end(), mStat.training_query_counter_h.begin());
        //printf("VoxelNetwork::reduce to get total counts\n");
        mStat.inference_query_count = thrust::reduce(mStat.inference_query_counter_h.begin(), mStat.inference_query_counter_h.end(), 0, thrust::plus<uint32_t>());
        mStat.training_query_count = thrust::reduce(mStat.training_query_counter_h.begin(), mStat.training_query_counter_h.end(), 0, thrust::plus<uint32_t>());
        mStat.training_sample_count = thrust::reduce(mStat.training_sample_counter_h.begin(), mStat.training_sample_counter_h.end(), 0, thrust::plus<uint32_t>());
        //printf("VoxelNetwork::sorting inference queries by voxel index\n");
        mStat.inference_query_voxel.resize(mStat.inference_query_count);
        thrust::transform(mResource.inferenceQuery, mResource.inferenceQuery + mStat.inference_query_count,
            mStat.inference_query_voxel.begin(), get_voxel_index());
        // until here, the all above operations costs < 1ms

        // strange, the transform costs ~0.2ms while the sort_by_key costs 10ms (Aris: ! ? ! ?)
        //printf("VoxelNetwork::thrust sorting %d inference queries\n", mStat.inference_query_voxel.size());
        // copy these queries costs ~1ms!
        {
            //thrust::copy(mResource.inferenceQuery, mResource.inferenceQuery + mStat.inference_query_count, mStat.inference_query.begin());
            //thrust::sort_by_key(mStat.inference_query_voxel.begin(), mStat.inference_query_voxel.end(), mStat.inference_query.begin(), thrust::less<uint32_t>());
            thrust::sort_by_key(mStat.inference_query_voxel.begin(), mStat.inference_query_voxel.end(), mResource.inferenceQuery, thrust::less<uint32_t>());
        }
        thrust::exclusive_scan(mStat.inference_query_counter_h.begin(), mStat.inference_query_counter_h.end(), mStat.inference_query_offset.begin());
        //printf("VoxelNetwork::preparing finished for the current frame\n");

        // store training sample into corresponding voxel's storage. this takes ~4ms.
        linear_kernel(sortForTraining<float>, 0, inference_stream, mStat.training_sample_count,
            mResource.trainingSample, mMemory->sample_voxel_counter.data().get(), mMemory->sample_storage.data().get(), voxel_param.voxel_num, max_training_sample_voxel);
        thrust::copy(mMemory->sample_voxel_counter.begin(), mMemory->sample_voxel_counter.end(), mMemory->sample_voxel_counter_h.begin());
        //for (int i = 0; i < voxel_param.voxel_num; i++) 
            //    printf("VoxelNetwork::voxel #%d now has %d training samples\n", i, mMemory->sample_voxel_counter_h[i]);
        
        // regenerate random sequence for training sample shuffling
        curandGenerateUniform(rng, mMemory->random_seq->data(), max_training_sample_voxel);
    }

    void VoxelNetwork::inference()
    {
        //uint32_t n_elements = mCounter->inference_query_count;
        uint32_t n_elements = mStat.inference_query_count;
        uint32_t n_voxels = voxel_param.voxel_num;
        if (!n_elements) return;

        for (int i = 0; i < n_voxels; i++) {
            uint32_t n_query = mStat.inference_query_counter_h[i];
            uint32_t inference_batch_size = next_multiple(n_query, 128u);
            if (n_query == 0) continue;

            RadianceQuery* query_data_ptr= mResource.inferenceQuery + mStat.inference_query_offset[i];
            linear_kernel(generateInferenceData<input_dim>, 0, inference_stream, n_query,
                query_data_ptr, mMemory->inference_data->data());
            float* input_data_ptr = mMemory->inference_data->data();
            float3* output_data_ptr = (float3*)mMemory->inference_target->data() + mStat.inference_query_offset[i];

            GPUMatrix input_data((float*)input_data_ptr, input_dim, inference_batch_size);
            GPUMatrix output_data((float*)output_data_ptr, output_dim, inference_batch_size);
            //printf("Submitting voxel #%d inference task: offset %d, and n#queries %d\n",
            //    i, mStat.inference_query_offset[i], n_query);
            mNetwork->voxel_network[i]->inference(inference_stream, input_data, output_data);
        }
        linear_kernel(mapRadianceToScreen<float>, 0, inference_stream, n_elements,
            mResource.inferenceQuery, mMemory->inference_target->data(), mResource.screenResult);
        CUDA_CHECK_THROW(cudaStreamSynchronize(inference_stream));
    }

    void VoxelNetwork::train(float& loss)
    {
        loss = 0;
        float current_loss = 0.;
        uint32_t n_all_samples = 0;
        uint32_t n_voxels = voxel_param.voxel_num;
        for (int i = 0; i < n_voxels; i++) {
            if (mStat.training_sample_counter_h[i] >= 128) {
                uint32_t training_batch_size = std::min(previous_multiple(mStat.training_sample_counter_h[i], 128u), batch_size);
                uint32_t total_samples = std::min(mMemory->sample_voxel_counter_h[i], (uint64_t)max_training_sample_voxel);
                n_all_samples += training_batch_size;
                auto s = mMemory->sample_storage[i];
                linear_kernel(generateTrainingData<input_dim>, 0, training_stream, training_batch_size,
                    mMemory->sample_storage[i], total_samples,
                    mMemory->training_data->data(), mMemory->training_target->data(), mMemory->random_seq->data());
                mMemory->training_data->set_size(input_dim, training_batch_size);
                mMemory->training_target->set_size(output_dim, training_batch_size);
                //printf("VoxelNetwork::training spawn %d training samples for voxel network #%d\n", training_batch_size, i);
                mNetwork->voxel_trainer[i]->training_step(training_stream, *mMemory->training_data, *mMemory->training_target, &current_loss);
                loss += current_loss * training_batch_size;
            }
        }
        loss /= n_all_samples;
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
