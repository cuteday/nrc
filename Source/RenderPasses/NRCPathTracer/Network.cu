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

    //using network_precision_t = float;

    // 4 steps each frame, with 16384 samples per batch
    int resolution = 1920 * 1080;
    int padded_resolution = next_multiple(resolution, 256);
    int batch_size = 1 << 14;
    const int input_dim = 5;         // pos, dir
    const int output_dim = 3;        // RGB
    const int alignment = 16;         // input dim alignment
    std::string config_path = "../RenderPasses/NRCPathTracer/Data/default_nrc.json";

    // cuda related
    cudaStream_t inference_stream;
    cudaStream_t training_stream;

    struct {
        //using precision_t = network_precision_t;
        std::shared_ptr<Loss<precision_t>> loss = nullptr;
        std::shared_ptr<Optimizer<precision_t>> optimizer = nullptr;
        std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = nullptr;
        std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer = nullptr;
        //std::shared_ptr<Network<precision_t>> network = nullptr;
        //std::shared_ptr<Encoding<precision_t>> encoding = nullptr;

    }mNetwork;

    struct {
        // the GPUMatrix class supports MxN matrices only
        // the GPUMatrix store in a continuous area in memory, either row major or column major
        GPUMatrix<float>* train_data = nullptr;
        GPUMatrix<float>* train_target = nullptr;
        GPUMatrix<float>* pred_target = nullptr;
        GPUMatrix<float>* inference_data = nullptr;
        GPUMatrix<float>* inference_target = nullptr;
    }mMemory;
}

// kernels
template <uint32_t stride, typename T = float>
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

namespace NRC {
    NRCNetwork::NRCNetwork()
    {
        CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
        CUDA_CHECK_THROW(cudaStreamCreate(&training_stream));

        initializeNetwork();
    }

    void NRCNetwork::initializeNetwork()
    {
        //initialize network
        //using precision_t = network_precision_t;

        //logInfo("Loading custom network config at" + mNetworkParams.config_path);
        std::ifstream f(config_path);
        tcnn::json config = tcnn::json::parse(f, nullptr, true, true);

        json loss_opts = config.value("loss", json::object());
        json optimizer_opts = config.value("optimizer", json::object());
        json network_opts = config.value("network", json::object());
        json encoding_opts = config.value("encoding", json::object());

        mNetwork.loss = std::shared_ptr<Loss<precision_t>>(create_loss<precision_t>(loss_opts) );
        mNetwork.optimizer = std::shared_ptr<Optimizer<precision_t>>(create_optimizer<precision_t>(optimizer_opts));
        //mNetwork.network = std::shared_ptr<Network<precision_t>>(create_network<precision_t>(network_opts));
        mNetwork.network = std::make_shared<NetworkWithInputEncoding<precision_t>>(input_dim, 0, output_dim, encoding_opts, network_opts);
        mNetwork.trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(mNetwork.network, mNetwork.optimizer, mNetwork.loss);

        mMemory.train_data = new GPUMatrix<float>(input_dim, batch_size);
        mMemory.train_target = new GPUMatrix<float>(output_dim, batch_size);
        mMemory.pred_target = new GPUMatrix<float>(output_dim, batch_size);
        mMemory.inference_data = new GPUMatrix<float>(input_dim, padded_resolution);
        mMemory.inference_target = new GPUMatrix<float>(output_dim, padded_resolution);
    }

    void NRCNetwork::inference(RadianceQuery* queries, int n_elements)
    {
        int n_batches = div_round_up(n_elements, batch_size);
        int n_queries = next_multiple(n_elements, 256);
        //for (int i = 0; i < 1; i++) {
        //    linear_kernel(generateBatchSequential<output_dim>, 0, inference_stream, batch_size,
        //        i * batch_size, queries, mMemory.train_data->data());
        //    mNetwork.network->inference(inference_stream, *mMemory.train_data, *mMemory.train_target);
        //}
        // this costs about < 1ms
        linear_kernel(generateBatchSequential<output_dim>, 0, inference_stream, n_elements,
            0, queries, mMemory.inference_data->data());
        cudaStreamSynchronize(inference_stream);
        mNetwork.network->inference(inference_stream, *mMemory.inference_data, *mMemory.inference_target);
        cudaStreamSynchronize(inference_stream);
    }

    void NRCNetwork::train(float& loss)
    {
        mNetwork.trainer->training_step(training_stream, *mMemory.train_data, *mMemory.train_target, &loss);
        cudaStreamSynchronize(training_stream);
    }
}
