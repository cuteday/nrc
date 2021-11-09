#include "NRC.h"



namespace NRC {

    NRCInterface::NRCInterface(){
        if (!FalcorCUDA::initCUDA()) {
            Falcor::logError("Cuda init failed");
            return;
        }
        network = NRCNetwork::SharedPtr(new NRCNetwork());
    }

}
