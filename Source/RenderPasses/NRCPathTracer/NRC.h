#pragma once
#include <cuda_runtime.h>

#include <Falcor.h>

#include "Network.h"
#include "FalcorCUDA.h"


using Falcor::uint;
using Falcor::uint2;
using Falcor::uint3;
using Falcor::uint4;
using Falcor::float2;
using Falcor::float3;
using Falcor::float4;

namespace NRC {

    class NRCInterface {

    public:
        using SharedPtr = std::shared_ptr<NRCInterface>;
        using WeakPtr = std::weak_ptr<NRCInterface>;
        using SharedConstPtr = std::shared_ptr<const NRCInterface>;

        NRCInterface();

        void trainFrame();

        void inferenceFrame();

        void registerNRCResources(Falcor::Buffer::SharedPtr pScreenQueryBuffer,
            Falcor::Texture::SharedPtr pScreenResultTexture) {
            mParameters.screenSize = uint2(pScreenResultTexture->getWidth(), pScreenResultTexture->getHeight());
            mFalcorResources.screenQuery = (NRC::RadianceQuery*)FalcorCUDA::importResourceToDevicePointer(pScreenQueryBuffer);
            mFalcorResources.screenResult = FalcorCUDA::mapTextureToSurfaceObject(pScreenResultTexture, cudaArrayColorAttachment);
        }      

    private:
        NRCNetwork::SharedPtr network = nullptr;

        struct {
            uint2 screenSize;
        }mParameters;

        // register interop texture/surface here
        struct {
            NRC::RadianceQuery* screenQuery;
            cudaSurfaceObject_t screenResult;
        }mFalcorResources;
    };
}
