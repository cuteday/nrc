#include "NRC.h"

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
        network->train(loss);
    }

    void NRCInterface::inferenceFrame()
    {
        int n_queries = mParameters.screenSize.x * mParameters.screenSize.y;
        network->inference(mFalcorResources.screenQuery, mFalcorResources.screenResult,
            mParameters.screenSize.x, mParameters.screenSize.y);
    }

}
