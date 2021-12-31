#include "json/json.hpp"

#include "NRC.h"
#include "Parameters.h"

using namespace Falcor;
using namespace NRC::Parameters;

namespace NRC::Parameters {
    VoxelConfig voxel_param;
    
}

namespace NRC {
    NRCVoxelInterface::NRCVoxelInterface() {
        if (!FalcorCUDA::initCUDA()) {
            Falcor::logFatal("Cuda init failed");
            return;
        }
        logInfo("NRCVoxelInterface::working directory: " + std::filesystem::current_path().string());
        logInfo("NRCVoxelInterface::creating and initializing network");
        json config = json::parse(std::ifstream(config_path), nullptr, true, true);
        logInfo("NRCInterface::read config file from: " + config_path);
        
        // voxel parameters
        json voxel_config = config.value("voxel", json::object());
        json net_config = config.value("net", json::object());
        json voxel_size = voxel_config.value("size", R"([1,1,1])"_json);
        voxel_param.voxel_size = { voxel_size[0],voxel_size[1],voxel_size[2] };
        voxel_param.voxel_num = voxel_param.voxel_size[0] * voxel_param.voxel_size[1] * voxel_param.voxel_size[2];
        std::cout << "Voxel size set to: [ " << voxel_param.voxel_size[0] << ", "
            << voxel_param.voxel_size[1] << ", "
            << voxel_param.voxel_size[2] << " ]" << std::endl;

        mNetwork = VoxelNetwork::SharedPtr(new VoxelNetwork(net_config));
    }

    void NRCVoxelInterface::prepare()
    {
        mNetwork->prepare();
    }

    void NRCVoxelInterface::trainFrame()
    {
        
        float loss;
        mNetwork->train(loss);
        mStats.n_frames++;
        mStats.training_loss_avg = mStats.ema_factor * mStats.training_loss_avg + (1 - mStats.ema_factor) * loss;

        if (mStats.n_frames % mStats.print_every == 0) {
            printStats();
        }
    }

    void NRCVoxelInterface::inferenceFrame()
    {
        mNetwork->inference();
    }

    void NRCVoxelInterface::printStats()
    {
        std::stringstream ss;
        ss << "Current frame: " << mStats.n_frames << "loss: " << mStats.training_loss_avg;
        Falcor::logInfo(ss.str());
        mNetwork->debug();
    }

    void NRCVoxelInterface::resetParameters()
    {
        mNetwork->reset();
    }
}
