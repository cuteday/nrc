#ifndef __NVCC__        // defined by nvcc complier 
#define __NVCC__
#endif

#include "Network.h"
#include "Helpers.h"
#include "Parameters.h"

#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>

using namespace tcnn;
using precision_t = tcnn::network_precision_t;

#define GPUMatrix GPUMatrix<float, CM>

namespace {

    // 4 steps each frame, with 16384 samples per batch
    uint32_t resolution = 1920 * 1080;    // is a multiple of 256
    const uint32_t batch_size = 1 << 14;
    const uint32_t n_train_batch = 4;
    const uint32_t self_query_batch_size = 1 << 16;     // ~ 57600
#if AUX_INPUTS
    const uint32_t input_dim = 14;         // pos dir normal roughness diffuse specular
#else
    const uint32_t input_dim = 5;           // pos, dir
#endif
    const uint32_t output_dim = 3;        // RGB
    //const uint32_t alignment = 16;        // input dim alignment
    const std::string config_path = "../RenderPasses/NRCPathTracer/Data/default_nrc_ema.json";

    // cuda related
    cudaStream_t inference_stream;
    cudaStream_t training_stream;
    curandGenerator_t rng;

    struct _Network { 
        std::shared_ptr<Loss<precision_t>> loss = nullptr;
        std::shared_ptr<Optimizer<precision_t>> optimizer = nullptr;
        std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = nullptr;
        std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer = nullptr;
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

    _Memory* mMemory;
    _Network* mNetwork;
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
        *(float3*)&training_target[output_index] = radiance;
    }
}

template <typename T = float>
__global__ void mapPredRadianceToScreen(uint32_t n_elements, uint32_t width,
    NRC::RadianceQuery* queries, T* data, cudaSurfaceObject_t output) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = i % width, y = i / width;
    uint32_t index = i * 3;
    float3 radiance = { data[index + 0] , data[index + 1], data[index + 2] };

#if REFLECTANCE_FACT
    //radiance = radiance * (queries[i].diffuse + queries[i].specular);
#endif
    float4 val = { radiance.x, radiance.y, radiance.z, 1.0f };
    surf2Dwrite(val, output, (int)sizeof(float4) * x, y);
}

template <class T>
__global__ void mapPredRadianceToScreen2(NRC::RadianceQuery* queries, T* data, cudaSurfaceObject_t output,
    uint32_t width, uint32_t height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uint32_t index = y * width + x;
        uint32_t data_index = index * 3;
        float3 radiance = { data[data_index + 0], data[data_index + 1], data[data_index + 2]};

#if REFLECTANCE_FACT
        //radiance = radiance * (queries[index].diffuse + queries[index].specular);
#endif
        float4 val = { radiance.x, radiance.y, radiance.z, 1.0f };
        surf2Dwrite(val, output, (int)sizeof(float4) * x, y);
    }
}

template <typename T = float>
__global__ void chkNaN(uint32_t n_elements, T* data) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n_elements) return;
    if (isnan(data[i]) || isinf(data[i])) {
        data[i] = (T)0.f;
    }
}

namespace NRC {
    NRCNetwork::NRCNetwork()
    {
        CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
        CUDA_CHECK_THROW(cudaStreamCreate(&training_stream));
        training_stream = inference_stream;

        CURAND_CHECK_THROW(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK_THROW(curandSetPseudoRandomGeneratorSeed(rng, 7272ULL));
        CURAND_CHECK_THROW(curandSetStream(rng, training_stream));

        initializeNetwork();
    }

    NRCNetwork::~NRCNetwork()
    {
        delete mNetwork;
        delete mMemory;
    }

    void NRCNetwork::initializeNetwork()
    {
        mNetwork = new _Network();
        mMemory = new _Memory();

        //initialize network
        std::ifstream f(config_path);
        tcnn::json config = tcnn::json::parse(f, nullptr, true, true);

        json loss_opts = config.value("loss", json::object());
        json optimizer_opts = config.value("optimizer", json::object());
        json network_opts = config.value("network", json::object());
        json encoding_opts = config.value("encoding", json::object());

        mNetwork->loss = std::shared_ptr<Loss<precision_t>>(create_loss<precision_t>(loss_opts) );
        mNetwork->optimizer = std::shared_ptr<Optimizer<precision_t>>(create_optimizer<precision_t>(optimizer_opts));
#if AUX_INPUTS
        mNetwork->network = std::make_shared<NetworkWithInputEncoding<precision_t>>(8, 6, output_dim, encoding_opts, network_opts);
#else
        mNetwork->network = std::make_shared<NetworkWithInputEncoding<precision_t>>(input_dim, 0, output_dim, encoding_opts, network_opts);
#endif
        mNetwork->trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(mNetwork->network, mNetwork->optimizer, mNetwork->loss);

        learning_rate = mNetwork->optimizer->learning_rate();
        mMemory->training_data = new GPUMatrix(input_dim, batch_size);
        mMemory->training_target = new GPUMatrix(output_dim, batch_size);
        mMemory->inference_data = new GPUMatrix(input_dim, resolution);
        mMemory->inference_target = new GPUMatrix(output_dim, resolution);
        mMemory->training_self_query = new GPUMatrix(input_dim, self_query_batch_size);
        mMemory->training_self_pred = new GPUMatrix(output_dim, self_query_batch_size);

        mMemory->random_seq = new GPUMemory<float>(n_train_batch * batch_size);
        CURAND_CHECK_THROW(curandGenerateUniform(rng, mMemory->random_seq->data(), n_train_batch * batch_size));
    }

    void NRCNetwork::reset()
    {
        cudaStreamSynchronize(training_stream);
        cudaStreamSynchronize(inference_stream);
        mNetwork->trainer->initialize_params(seed);
    }

    void NRCNetwork::inference(RadianceQuery* queries, cudaSurfaceObject_t output,
        uint32_t width, uint32_t height)
    {
        uint32_t n_elements = width * height;

        // this input generation process takes about ~1ms.
        linear_kernel(generateBatchSequential<input_dim>, 0, inference_stream, n_elements,
            0, queries, mMemory->inference_data->data());
        
        mNetwork->network->inference(inference_stream, *mMemory->inference_data, *mMemory->inference_target);
        //linear_kernel(mapPredRadianceToScreen<float>, 0, inference_stream, n_elements, width, queries, mMemory->inference_target->data(), output);

        dim3 dimBlock(16, 16), dimGrid(div_round_up(width, 16u), div_round_up(height, 16u));
        mapPredRadianceToScreen2<float> <<<dimGrid, dimBlock, 0, inference_stream >>>
            (queries, mMemory->inference_target->data(), output, width, height);

        cudaStreamSynchronize(inference_stream);
    }

    void NRCNetwork::train(RadianceQuery* self_queries, uint32_t* self_query_counter,
        RadianceSample* training_samples, uint32_t* training_sample_counter, float& loss)
    {
        // setup change-able parameters
        mNetwork->optimizer->set_learning_rate(learning_rate);

        // self query
        linear_kernel(generateBatchSequential<input_dim>, 0, training_stream, self_query_batch_size,
            0, self_queries, mMemory->training_self_query->data());
        mNetwork->network->inference(training_stream, *mMemory->training_self_query, *mMemory->training_self_pred);

        // training
#if 1   // randomly select 4 training batches over all samples
        CURAND_CHECK_THROW(curandGenerateUniform(rng, mMemory->random_seq->data(), n_train_batch * batch_size));
        for (uint32_t i = 0; i < n_train_batch; i++) {
            linear_kernel(generateTrainingDataFromSamples<input_dim, float>, 0, training_stream, batch_size,
                i * batch_size, training_samples, self_queries, mMemory->training_self_pred->data(),
                mMemory->training_data->data(), mMemory->training_target->data(),
                training_sample_counter, self_query_counter, mMemory->random_seq->data());
            mNetwork->trainer->training_step(training_stream, *mMemory->training_data, *mMemory->training_target, &loss);
        }
#elif 0   // batched training over all samples
        // get sample count...
        // TODO: this costs ~1.5ms. find a better strategy.
        uint32_t sample_count;
        cudaMemcpy(&sample_count, training_sample_counter, sizeof(float), cudaMemcpyDeviceToHost);

        for (uint32_t i = 0; i + batch_size <= sample_count; i += batch_size) {
            linear_kernel(generateTrainingDataFromSamples<input_dim, float>, 0, training_stream, batch_size,
                i, training_samples, mMemory->training_self_pred->data(),
                mMemory->training_data->data(), mMemory->training_target->data(),
                training_sample_counter, self_query_counter);

            // not here: we check NaNs and INFs in the shader before adding the training samples.
            //linear_kernel(chkNaN<float>, 0, training_stream, mMemory->training_data->n_elements(), mMemory->training_data->data());
            //linear_kernel(chkNaN<float>, 0, training_stream, mMemory->training_target->n_elements(), mMemory->training_target->data());

            mNetwork->trainer->training_step(training_stream, *mMemory->training_data, *mMemory->training_target, &loss);
            std::cout << "Loss at current step: " << loss << std::endl;
        }
#elif 0 // single batched training
        linear_kernel(generateTrainingDataFromSamples<input_dim, float>, 0, training_stream, batch_size,
            0, training_samples, mMemory->training_self_pred->data(),
            mMemory->training_data->data(), mMemory->training_target->data(),
            training_sample_counter, self_query_counter);

        linear_kernel(chkNaN<float>, 0, training_stream, mMemory->training_data->n_elements(), mMemory->training_data->data());
        linear_kernel(chkNaN<float>, 0, training_stream, mMemory->training_target->n_elements(), mMemory->training_target->data());

        mNetwork->trainer->training_step(training_stream, *mMemory->training_data, *mMemory->training_target, &loss);
        std::cout << "Loss at current step: " << loss << std::endl;
#endif
        cudaStreamSynchronize(training_stream);
    }
}
