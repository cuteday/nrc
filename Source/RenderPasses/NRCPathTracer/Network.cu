#ifndef __NVCC__        // defined by nvcc complier 
#define __NVCC__
#endif

#include "Network.h"

#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>

using namespace tcnn;
using precision_t = tcnn::network_precision_t;

#define GPUMatrix GPUMatrix<float, CM>
//typedef tcnn::GPUMatrix<float, RM> GPUMatrix;
//using GPUMatrix = tcnn::GPUMatrix<float, RM>

namespace {

    // 4 steps each frame, with 16384 samples per batch
    unsigned int resolution = 1920 * 1080;    // is a multiple of 256
    const unsigned int batch_size = 1 << 14;
    const unsigned int self_query_batch_size = 1 << 16;     // ~ 57600
    const unsigned int input_dim = 5;         // pos, dir
    const unsigned int output_dim = 3;        // RGB
    //const unsigned int alignment = 16;        // input dim alignment
    const std::string config_path = "../RenderPasses/NRCPathTracer/Data/default_nrc.json";

    // cuda related
    cudaStream_t inference_stream;
    cudaStream_t training_stream;

    struct _Network { 
        std::shared_ptr<Loss<precision_t>> loss = nullptr;
        std::shared_ptr<Optimizer<precision_t>> optimizer = nullptr;
        std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = nullptr;
        std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer = nullptr;
        //std::shared_ptr<Network<precision_t>> network = nullptr;
        //std::shared_ptr<Encoding<precision_t>> encoding = nullptr;
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
    };

    _Memory* mMemory;
    _Network* mNetwork;
}

// device code helper functions
template <typename T = float3>
__device__ T vec3_mult(T a, T b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

template <typename T = float3>
__device__ T vec3_add(T a, T b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

// linear kernels with only x-dim not 1. must be called using linear_kernal()
// blockDim = 128, threadIdx is the index of a thread within a thread block, i.e. in [0, 128)
// reference linear_kernel() for details.
// stride: input dim
template <uint32_t stride, typename T = float>
__global__ void generateBatchSequential(uint32_t n_elements, uint32_t offset, 
    NRC::RadianceQuery* __restrict__ queries, T* __restrict__ data) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset < n_elements) {
        uint32_t data_index = i * stride, query_index = i + offset;

        data[data_index + 0] = (T)queries[query_index].pos.x;
        data[data_index + 1] = (T)queries[query_index].pos.y;
        data[data_index + 2] = (T)queries[query_index].pos.z;
        data[data_index + 3] = (T)queries[query_index].dir.x;
        data[data_index + 4] = (T)queries[query_index].dir.y;
    }
}

template <typename T = float>
__global__ void generateTrainingDataFromSamples(uint32_t n_elements, uint32_t offset,
    NRC::RadianceSample* __restrict__ samples, T* __restrict__ self_query_pred,
    T* __restrict__ training_data, T* __restrict__ training_target,
    uint32_t* training_sample_counter, uint32_t* self_query_counter) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset > n_elements) return;
    uint32_t data_index = i * 5, sample_index = i + offset;
    uint32_t pred_index = samples[sample_index].idx >= 0 ? samples[sample_index].idx : 0;

    if (sample_index < *training_sample_counter && pred_index < *self_query_counter) {
        float3 factor = samples[sample_index].a, bias = samples[sample_index].b;
        uint32_t output_index = i * 3;

        training_data[data_index + 0] = (T)samples[sample_index].query.pos.x;
        training_data[data_index + 1] = (T)samples[sample_index].query.pos.y;
        training_data[data_index + 2] = (T)samples[sample_index].query.pos.z;
        training_data[data_index + 3] = (T)samples[sample_index].query.dir.x;
        training_data[data_index + 4] = (T)samples[sample_index].query.dir.y;

        float3 pred_radiance = { self_query_pred[pred_index], self_query_pred[pred_index + 1], self_query_pred[pred_index + 2] };
        float3 radiance = vec3_add(vec3_mult(pred_radiance, factor), bias);
        training_target[output_index + 0] = (T)radiance.x;
        training_target[output_index + 1] = (T)radiance.y;
        training_target[output_index + 2] = (T)radiance.z;
    }
}

template <typename T = float>
__global__ void mapPredRadianceToScreen(uint32_t n_elements, uint32_t width,
    T* __restrict__ data, cudaSurfaceObject_t output) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = i % width, y = i / width;
    unsigned int index = i * 3;
    float4 radiance = { data[index + 0] , data[index + 1], data[index + 2], 1.f };
    surf2Dwrite(radiance, output, (int)sizeof(float4) * x, y);
}

template <class T>
__global__ void mapPredRadianceToScreen2(T* __restrict__ data, cudaSurfaceObject_t output,
    unsigned int width, unsigned int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        unsigned int index = (y * width + x) * 3;
        float4 radiance = { data[index + 0], data[index + 1], data[index + 2], 1.f };
    
//         float greyScale = ((float)x / width) * ((float)y / height);
//        float4 radiance = { greyScale, greyScale, greyScale, 1.f };
        surf2Dwrite(radiance, output, (int)sizeof(float4) * x, y);
    }
}

template <typename T = float>
__global__ void chkNaN(uint32_t n_elements, T* __restrict__ data) {
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
        mNetwork->network = std::make_shared<NetworkWithInputEncoding<precision_t>>(input_dim, 0, output_dim, encoding_opts, network_opts);
        mNetwork->trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(mNetwork->network, mNetwork->optimizer, mNetwork->loss);

        mMemory->training_data = new GPUMatrix(input_dim, batch_size);
        mMemory->training_target = new GPUMatrix(output_dim, batch_size);
        mMemory->inference_data = new GPUMatrix(input_dim, resolution);
        mMemory->inference_target = new GPUMatrix(output_dim, resolution);
        mMemory->training_self_query = new GPUMatrix(input_dim, self_query_batch_size);
        mMemory->training_self_pred = new GPUMatrix(output_dim, self_query_batch_size);
    }

    void NRCNetwork::reset()
    {
        cudaStreamSynchronize(training_stream);
        cudaStreamSynchronize(inference_stream);
        mNetwork->trainer->initialize_params(seed);
    }

    void NRCNetwork::inference(RadianceQuery* queries, cudaSurfaceObject_t output,
        unsigned int width, unsigned int height)
    {

        unsigned int n_elements = width * height;
        //int n_batches = div_round_up(n_elements, batch_size);
        //int n_queries = next_multiple(n_elements, 256u);
        
        //std::cout << "Inference batch size: " << mMemory->inference_data->rows() << mMemory->inference_data->cols() << std::endl;

        // this input generation process takes about ~1ms.
        linear_kernel(generateBatchSequential<input_dim>, 0, inference_stream, n_elements,
            0, queries, mMemory->inference_data->data());
        
        mNetwork->network->inference(inference_stream, *mMemory->inference_data, *mMemory->inference_target);

        //linear_kernel(mapPredRadianceToScreen<float>, 0, inference_stream, n_elements, width, mMemory->inference_target->data(), output);

        dim3 dimBlock(16, 16), dimGrid(div_round_up(width, 16u), div_round_up(height, 16u));
        mapPredRadianceToScreen2<float> <<<dimGrid, dimBlock, 0, inference_stream >>>
            (mMemory->inference_target->data(), output, width, height);

        cudaStreamSynchronize(inference_stream);
    }

    void NRCNetwork::train(RadianceQuery* self_queries, uint32_t* self_query_counter,
        RadianceSample* training_samples, uint32_t* training_sample_counter, float& loss)
    {
        // self query
        linear_kernel(generateBatchSequential<input_dim>, 0, training_stream, self_query_batch_size,
            0, self_queries, mMemory->training_self_query->data());

        mNetwork->network->inference(training_stream, *mMemory->training_self_query, *mMemory->training_self_pred);

        // training
        linear_kernel(generateTrainingDataFromSamples<float>, 0, training_stream, batch_size,
            0, training_samples, mMemory->training_self_pred->data(),
            mMemory->training_data->data(), mMemory->training_target->data(),
            training_sample_counter, self_query_counter);
        linear_kernel(chkNaN<float>, 0, training_stream, mMemory->training_data->n_elements(), mMemory->training_data->data());
        linear_kernel(chkNaN<float>, 0, training_stream, mMemory->training_target->n_elements(), mMemory->training_target->data());
        mNetwork->trainer->training_step(training_stream, *mMemory->training_data, *mMemory->training_target, &loss);
        cudaStreamSynchronize(training_stream);
        std::cout << "Loss at current step: " << loss << std::endl;
    }
}
