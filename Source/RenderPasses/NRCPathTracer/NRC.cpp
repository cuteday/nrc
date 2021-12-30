#include "NRC.h"
#include "Parameters.h"

using namespace Falcor;

namespace NRC {

    NRCInterface::NRCInterface(){
        if (!FalcorCUDA::initCUDA()) {
            Falcor::logFatal("Cuda init failed");
            return;
        }
        logInfo("NRCInterface::working directory: " + std::filesystem::current_path().string());
        logInfo("NRCInferface::creating and initializing network");
        mNetwork = NRCNetwork::SharedPtr(new NRCNetwork());
    }

    void NRCInterface::beginFrame()
    {
        mNetwork->beginFrame(mFalcorResources.counterBufferPtr);
    }

    void NRCInterface::trainFrame()
    {
        float loss;
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
