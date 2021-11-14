#define __NVCC__

#include "Network.h"

#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>


using namespace tcnn;

namespace {

    //using network_precision_t = float;
    using network_precision_t = __half;
    // 4 steps each frame, with 16384 samples per batch
    uint32_t batch_size = 1 << 14;
    const int input_dims = 5;         // pos, dir
    const int output_dims = 3;        // RGB
    std::string config_path = "../RenderPasses/NRCPathTracer/Data/default_nrc.json";


    struct {
        using precision_t = network_precision_t;
        std::shared_ptr<Loss<precision_t>> loss = nullptr;
        std::shared_ptr<Optimizer<precision_t>> optimizer = nullptr;
        std::shared_ptr<Network<precision_t>> network = nullptr;

    }mNetwork;

    struct {
        // the GPUMatrix class supports MxN matrices only
        // the GPUMatrix store in a continuous area in memory, either row major or column major
        GPUMatrix<float>* train_data = nullptr;
        GPUMatrix<float>* train_target = nullptr;
        GPUMatrix<float>* pred_target = nullptr;
    }mNetworkMemory;
}

// kernels
template <uint32_t stride>
__global__ void generateBatchSequential(uint32_t n_elements, uint32_t offset, 
    NRC::RadianceQuery* __restrict__ queries, float* __restrict__ data) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n_elements)return;
    uint32_t index = (offset + i) * stride;

    data[index + 0] = queries[i].pos.x;
    data[index + 1] = queries[i].pos.y;
    data[index + 2] = queries[i].pos.z;
    data[index + 3] = queries[i].dir.x;
    data[index + 4] = queries[i].dir.y;
}

namespace NRC {
    NRCNetwork::NRCNetwork()
    {
        initializeNetwork();
    }

    void NRCNetwork::initializeNetwork()
    {
        //initialize network
        using precision_t = network_precision_t;

        //logInfo("Loading custom network config at" + mNetworkParams.config_path);
        std::ifstream f(config_path);
        tcnn::json config = tcnn::json::parse(f, nullptr, true, true);

        json loss_opts = config.value("loss", json::object());
        json optimizer_opts = config.value("optimizer", json::object());
        json network_opts = config.value("network", json::object());

        mNetwork.loss = std::shared_ptr<Loss<precision_t>>(create_loss<precision_t>(loss_opts) );
        mNetwork.optimizer = std::shared_ptr<Optimizer<precision_t>>(create_optimizer<precision_t>(optimizer_opts));
        //mNetwork.network = std::shared_ptr<Network<precision_t>>(create_network<precision_t>(network_opts));

        mNetworkMemory.train_data = new GPUMatrix<float>(input_dims, batch_size);
        mNetworkMemory.train_target = new GPUMatrix<float>(output_dims, batch_size);
        mNetworkMemory.train_data = new GPUMatrix<float>(output_dims, batch_size);
    }

    void NRCNetwork::inference(RadianceQuery* queries, uint32_t n_elements)
    {
        int n_batches = div_round_up(n_elements, batch_size);
        for (int i = 0; i < n_batches; i++) {
            linear_kernel(generateBatchSequential<output_dims>, 0, nullptr, batch_size,
                i * batch_size, queries, mNetworkMemory.train_data);
        }
    }
}
