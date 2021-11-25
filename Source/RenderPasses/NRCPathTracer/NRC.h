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

        void printStats();

        void resetParameters();

        void registerNRCResources(Falcor::Buffer::SharedPtr pScreenQueryBuffer,
            Falcor::Texture::SharedPtr pScreenResultTexture,
            Falcor::Buffer::SharedPtr pTrainingQueryBuffer,
            Falcor::Buffer::SharedPtr pTrainingSampleBuffer,
            Falcor::Buffer::SharedPtr pSharedCounterBuffer) {

            mParameters.screenSize = uint2(pScreenResultTexture->getWidth(), pScreenResultTexture->getHeight());
            mFalcorResources.screenResult = FalcorCUDA::mapTextureToSurfaceObject(pScreenResultTexture, cudaArrayColorAttachment);
            mFalcorResources.screenQuery = (NRC::RadianceQuery*)pScreenQueryBuffer->getCUDADeviceAddress();
            mFalcorResources.trainingQuery = (NRC::RadianceQuery*)pTrainingQueryBuffer->getCUDADeviceAddress();
            mFalcorResources.trainingSample = (NRC::RadianceSample*)pTrainingSampleBuffer->getCUDADeviceAddress();
            uint32_t* counterBuffer = (uint32_t*)pSharedCounterBuffer->getCUDADeviceAddress();
            /*mFalcorResources.screenQuery = (NRC::RadianceQuery*)FalcorCUDA::importResourceToDevicePointer(pScreenQueryBuffer);
            mFalcorResources.trainingQuery = (NRC::RadianceQuery*)FalcorCUDA::importResourceToDevicePointer(pTrainingQueryBuffer);
            mFalcorResources.trainingSample = (NRC::RadianceSample*)FalcorCUDA::importResourceToDevicePointer(pTrainingSampleBuffer);
            uint32_t* counterBuffer = (uint32_t*)FalcorCUDA::importResourceToDevicePointer(pSharedCounterBuffer);*/
            mFalcorResources.trainingQueryCounter = &counterBuffer[0];
            mFalcorResources.trainingSampleCounter = &counterBuffer[1];
        }      

//    private:
        NRCNetwork::SharedPtr network = nullptr;
        struct {
            int n_frames = 0;
            float training_loss_avg = 0;    // EMA
            const float ema_factor = 0.8f;
            const int print_every = 100;
        }mStats;

        struct {
            uint2 screenSize;
        }mParameters;

        // register interop texture/surface here
        struct {
            // cuda device pointers in unified memory space.
            NRC::RadianceQuery* screenQuery = nullptr;           
            cudaSurfaceObject_t screenResult;       // write inferenced results here
            NRC::RadianceQuery* trainingQuery = nullptr;
            NRC::RadianceSample* trainingSample = nullptr;
            uint32_t* trainingQueryCounter = nullptr;
            uint32_t* trainingSampleCounter = nullptr;
        }mFalcorResources;
    };
}
