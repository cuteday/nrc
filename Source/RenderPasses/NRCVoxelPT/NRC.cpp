#include "json/json.hpp"

#include "NRC.h"
#include "Parameters.h"

using namespace Falcor;
using namespace NRC::Parameters;


namespace NRC::Parameters {
    VoxelConfig voxel_param;
    
}

namespace NRC {
    NRCInterface::NRCInterface() {
        if (!FalcorCUDA::initCUDA()) {
            Falcor::logFatal("Cuda init failed");
            return;
        }
        logInfo("NRCInterface::working directory: " + std::filesystem::current_path().string());
        logInfo("NRCInterface::creating and initializing network");
        mNetwork = NRCNetwork::SharedPtr(new NRCNetwork());
    }

    void NRCInterface::beginFrame()
    {
        mNetwork->beginFrame(mFalcorResources.counterBufferPtr);
    }

    void NRCInterface::trainFrame()
    {
        float loss;
        /*mNetwork->train(mFalcorResources.trainingQuery, Parameters::max_training_query_size,
            mFalcorResources.trainingSample, Parameters::max_training_sample_size, loss);*/
        mNetwork->train(mFalcorResources.trainingQuery, mFalcorResources.trainingQueryCounter,
            mFalcorResources.trainingSample, mFalcorResources.trainingSampleCounter, loss);
        mStats.n_frames++;
        mStats.training_loss_avg = mStats.ema_factor * mStats.training_loss_avg + (1 - mStats.ema_factor) * loss;

        if (mStats.n_frames % mStats.print_every == 0) {
            printStats();
        }
    }

    void NRCInterface::inferenceFrame()
    {
        //mNetwork->inference(mFalcorResources.screenQuery, mFalcorResources.screenResult,
        //    mParameters.screenSize.x, mParameters.screenSize.y);
        mNetwork->inference(mFalcorResources.screenQuery, mFalcorResources.inferenceQueryPixel, mFalcorResources.screenResult);
    }

    void NRCInterface::printStats()
    {
        std::stringstream ss;
        ss << "Current frame: " << mStats.n_frames << "loss: " << mStats.training_loss_avg;
        Falcor::logInfo(ss.str());
    }

    void NRCInterface::resetParameters()
    {
        mNetwork->reset();
    }
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
        std::cout << "Voxel size set to: [ " << voxel_param.voxel_size[0] << ", "
            << voxel_param.voxel_size[1] << ", "
            << voxel_param.voxel_size[2] << " ]" << std::endl;

        mNetwork = VoxelNetwork::SharedPtr(new VoxelNetwork(net_config));
    }

    void NRCVoxelInterface::beginFrame()
    {
        mNetwork->beginFrame(mFalcorResources.counterBufferPtr);
    }

    void NRCVoxelInterface::trainFrame()
    {
        float loss;
        /*mNetwork->train(mFalcorResources.trainingQuery, Parameters::max_training_query_size,
            mFalcorResources.trainingSample, Parameters::max_training_sample_size, loss);*/
        mNetwork->train(mFalcorResources.trainingQuery, mFalcorResources.trainingQueryCounter,
            mFalcorResources.trainingSample, mFalcorResources.trainingSampleCounter, loss);
        mStats.n_frames++;
        mStats.training_loss_avg = mStats.ema_factor * mStats.training_loss_avg + (1 - mStats.ema_factor) * loss;

        if (mStats.n_frames % mStats.print_every == 0) {
            printStats();
        }
    }

    void NRCVoxelInterface::inferenceFrame()
    {
        //mNetwork->inference(mFalcorResources.screenQuery, mFalcorResources.screenResult,
        //    mParameters.screenSize.x, mParameters.screenSize.y);
        mNetwork->inference(mFalcorResources.screenQuery, mFalcorResources.inferenceQueryPixel, mFalcorResources.screenResult);
    }

    void NRCVoxelInterface::printStats()
    {
        std::stringstream ss;
        ss << "Current frame: " << mStats.n_frames << "loss: " << mStats.training_loss_avg;
        Falcor::logInfo(ss.str());
    }

    void NRCVoxelInterface::resetParameters()
    {
        mNetwork->reset();
    }
}
