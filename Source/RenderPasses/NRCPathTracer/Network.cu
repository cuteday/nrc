#ifndef __NVCC__        // defined by nvcc complier 
#define __NVCC__
#endif

#include "Network.h"

#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>


using namespace tcnn;
using precision_t = tcnn::network_precision_t;

namespace {

    // 4 steps each frame, with 16384 samples per batch
    unsigned int resolution = 1920 * 1080;
    unsigned int padded_resolution = next_multiple(resolution, 256u);
    const unsigned int batch_size = 1 << 14;
    const unsigned int input_dim = 5;         // pos, dir
    const unsigned int output_dim = 3;        // RGB
    const unsigned int alignment = 16;        // input dim alignment
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
        GPUMatrix<float>* train_data = nullptr;
        GPUMatrix<float>* train_target = nullptr;
        GPUMatrix<float>* pred_target = nullptr;
        GPUMatrix<float>* inference_data = nullptr;
        GPUMatrix<float>* inference_target = nullptr;
    };

    _Memory* mMemory;
    _Network* mNetwork;
}

// linear kernels with only x-dim not 1. must be called using linear_kernal()
template <uint32_t stride = 5, typename T = float>
__global__ void generateBatchSequential(uint32_t n_elements, uint32_t offset, 
    NRC::RadianceQuery* __restrict__ queries, T* __restrict__ data) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n_elements)return;
    uint32_t index = (offset + i) * stride;

    data[index + 0] = (T)queries[i].pos.x;
    data[index + 1] = (T)queries[i].pos.y;
    data[index + 2] = (T)queries[i].pos.z;
    data[index + 3] = (T)queries[i].dir.x;
    data[index + 4] = (T)queries[i].dir.y;
}

template <typename T = float>
__global__ void mapPredRadianceToScreen(uint32_t n_elements, uint32_t width,
    T* __restrict__ data, cudaSurfaceObject_t output) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = i % width, y = i / width;
    unsigned int index = i * 3;
    float4 radiance;
    radiance.x = data[index + 0];
    radiance.x = data[index + 1];
    radiance.x = data[index + 2];
    radiance.w = 0.0f;
    surf2Dwrite(radiance, output, sizeof(float4) * x, y);
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
        //using precision_t = network_precision_t;

        //logInfo("Loading custom network config at" + mNetworkParams.config_path);
        std::ifstream f(config_path);
        tcnn::json config = tcnn::json::parse(f, nullptr, true, true);

        json loss_opts = config.value("loss", json::object());
        json optimizer_opts = config.value("optimizer", json::object());
        json network_opts = config.value("network", json::object());
        json encoding_opts = config.value("encoding", json::object());

        mNetwork->loss = std::shared_ptr<Loss<precision_t>>(create_loss<precision_t>(loss_opts) );
        mNetwork->optimizer = std::shared_ptr<Optimizer<precision_t>>(create_optimizer<precision_t>(optimizer_opts));
        //mNetwork->network = std::shared_ptr<Network<precision_t>>(create_network<precision_t>(network_opts));
        mNetwork->network = std::make_shared<NetworkWithInputEncoding<precision_t>>(input_dim, 0, output_dim, encoding_opts, network_opts);
        mNetwork->trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(mNetwork->network, mNetwork->optimizer, mNetwork->loss);

        mMemory->train_data = new GPUMatrix<float>(input_dim, batch_size);
        mMemory->train_target = new GPUMatrix<float>(output_dim, batch_size);
        mMemory->pred_target = new GPUMatrix<float>(output_dim, batch_size);
        mMemory->inference_data = new GPUMatrix<float>(input_dim, padded_resolution);
        mMemory->inference_target = new GPUMatrix<float>(output_dim, padded_resolution);
    }

    void NRCNetwork::inference(RadianceQuery* queries, cudaSurfaceObject_t output,
        unsigned int width, unsigned int height)
    {
        unsigned int n_elements = width * height;
        int n_batches = div_round_up(n_elements, batch_size);
        int n_queries = next_multiple(n_elements, 256u);
        
        // this input generation process takes about ~1ms.
        linear_kernel(generateBatchSequential<output_dim>, 0, inference_stream, n_elements,
            0, queries, mMemory->inference_data->data());
        
        mNetwork->network->inference(inference_stream, *mMemory->inference_data, *mMemory->inference_target);

        linear_kernel(mapPredRadianceToScreen<float>, 0, inference_stream, n_elements, width, mMemory->inference_target->data(), output);
        cudaStreamSynchronize(inference_stream);
    }

    void NRCNetwork::train(float& loss)
    {
        mNetwork->trainer->training_step(training_stream, *mMemory->train_data, *mMemory->train_target, &loss);
        cudaStreamSynchronize(training_stream);
    }
}
