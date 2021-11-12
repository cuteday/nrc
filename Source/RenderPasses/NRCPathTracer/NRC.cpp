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

    void NRCInterface::trainFrame(Falcor::Buffer::SharedPtr pTrainingRadianceQueryBuffer, Falcor::Buffer::SharedPtr pTrainingRadianceRecordBuffer)
    {
    }

    void NRCInterface::inferenceFrame(Falcor::Buffer::SharedPtr pInferenceRadianceQueryTexture, Falcor::Texture::SharedPtr pScreenQueryFactorTexture, Falcor::Texture::SharedPtr pScreenQueryBiasTexture)
    {
    }

}
