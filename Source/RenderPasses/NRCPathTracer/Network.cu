#include "Network.h"

#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/config.h>

using namespace tcnn;

namespace {
    struct NetworkParams {
        //using network_precision_t = float;
        using network_precision_t = __half;
        // 4 steps each frame, with 16384 samples per batch
        int batch_size = 1 << 14;
        int input_dims = 5;
        int output_dims = 3;
        std::string config_path = "../RenderPasses/NRCPathTracer/Data/default_nrc.json";
    } mNetworkParams;

    struct {
        using precision_t = NetworkParams::network_precision_t;
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

namespace NRC {
    NRCNetwork::NRCNetwork()
    {
        initializeNetwork();
    }

    void NRCNetwork::initializeNetwork()
    {
        //initialize network
        using precision_t = NetworkParams::network_precision_t;

        //logInfo("Loading custom network config at" + mNetworkParams.config_path);
        std::ifstream f(mNetworkParams.config_path);
        tcnn::json config = tcnn::json::parse(f, nullptr, true, true);

        json loss_opts = config.value("loss", json::object());
        json optimizer_opts = config.value("optimizer", json::object());
        json network_opts = config.value("network", json::object());

        mNetwork.loss = std::shared_ptr<Loss<precision_t>>(create_loss<precision_t>(loss_opts) );
        mNetwork.optimizer = std::shared_ptr<Optimizer<precision_t>>(create_optimizer<precision_t>(optimizer_opts));
        //mNetwork.network = std::shared_ptr<Network<precision_t>>(create_network<precision_t>(network_opts));

        mNetworkMemory.train_data = new GPUMatrix<float>(mNetworkParams.input_dims, mNetworkParams.batch_size);
        mNetworkMemory.train_target = new GPUMatrix<float>(mNetworkParams.output_dims, mNetworkParams.batch_size);
        mNetworkMemory.train_data = new GPUMatrix<float>(mNetworkParams.output_dims, mNetworkParams.batch_size);

    }
}
