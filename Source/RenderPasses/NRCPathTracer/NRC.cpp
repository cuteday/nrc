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

}
