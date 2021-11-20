#include "NRC.h"
#include "Parameters.h"

using namespace Falcor;

namespace NRC {

    NRCInterface::NRCInterface(){
        if (!FalcorCUDA::initCUDA()) {
            Falcor::logError("Cuda init failed");
            return;
        }
        logInfo("NRCInterface::working directory: " + std::filesystem::current_path().string());
        logInfo("NRCInferface::creating and initializing network");
        network = NRCNetwork::SharedPtr(new NRCNetwork());
    }

    void NRCInterface::trainFrame()
    {
        float loss;
        /*network->train(mFalcorResources.trainingQuery, Parameters::max_training_query_size,
            mFalcorResources.trainingSample, Parameters::max_training_sample_size, loss);*/
        network->train(mFalcorResources.trainingQuery, mFalcorResources.trainingQueryCounter,
            mFalcorResources.trainingSample, mFalcorResources.trainingSampleCounter, loss);
        mStats.n_frames++;
        mStats.training_loss_avg = mStats.ema_factor * mStats.training_loss_avg + (1 - mStats.ema_factor) * loss;

        if (mStats.n_frames % mStats.print_every == 0) {
            printStats();
        }
    }

    void NRCInterface::inferenceFrame()
    {
        int n_queries = mParameters.screenSize.x * mParameters.screenSize.y;
        network->inference(mFalcorResources.screenQuery, mFalcorResources.screenResult,
            mParameters.screenSize.x, mParameters.screenSize.y);
    }
    void NRCInterface::printStats()
    {
        std::stringstream ss;
        ss << "Current frame: " << mStats.n_frames << "loss: " << mStats.training_loss_avg;
        Falcor::logInfo(ss.str());
    }
}
