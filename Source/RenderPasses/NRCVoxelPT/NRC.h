#pragma once

#include "VoxelNetwork.h"
#include "Network.h"
#include "FalcorCUDA.h"

#include <Falcor.h>
#include <cuda_runtime.h>

using Falcor::uint;
using Falcor::uint2;
using Falcor::uint3;
using Falcor::uint4;
using Falcor::float2;
using Falcor::float3;
using Falcor::float4;

namespace NRC {

    class NRCVoxelInterface {

    public:
        using SharedPtr = std::shared_ptr<NRCVoxelInterface>;
        using WeakPtr = std::weak_ptr<NRCVoxelInterface>;
        using SharedConstPtr = std::shared_ptr<const NRCVoxelInterface>;

        NRCVoxelInterface();

        void beginFrame();

        void trainFrame();

        void inferenceFrame();

        void printStats();

        void resetParameters();

        void registerNRCResources(Falcor::Buffer::SharedPtr pInferenceQueryBuffer,
            Falcor::Texture::SharedPtr pScreenResultTexture,
            Falcor::Buffer::SharedPtr pTrainingQueryBuffer,
            Falcor::Buffer::SharedPtr pTrainingSampleBuffer,
            Falcor::Buffer::SharedPtr pSharedCounterBuffer) {

            mParameters.screenSize = uint2(pScreenResultTexture->getWidth(), pScreenResultTexture->getHeight());
            mFalcorResources.screenResult = FalcorCUDA::mapTextureToSurfaceObject(pScreenResultTexture, cudaArrayColorAttachment);
            mFalcorResources.inferenceQuery = (NRC::RadianceQuery*)pInferenceQueryBuffer->getCUDADeviceAddress();
            mFalcorResources.trainingQuery = (NRC::RadianceQuery*)pTrainingQueryBuffer->getCUDADeviceAddress();
            mFalcorResources.trainingSample = (NRC::RadianceSample*)pTrainingSampleBuffer->getCUDADeviceAddress();
            uint32_t* counterBuffer = (uint32_t*)pSharedCounterBuffer->getCUDADeviceAddress();
            mFalcorResources.counterBufferPtr = counterBuffer;
        }

        //    private:
        VoxelNetwork::SharedPtr mNetwork = nullptr;

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
            NRC::RadianceQuery* inferenceQuery = nullptr;
            cudaSurfaceObject_t screenResult;       // write inferenced results here
            NRC::RadianceQuery* trainingQuery = nullptr;
            NRC::RadianceSample* trainingSample = nullptr;
            uint2* inferenceQueryPixel = nullptr;
            uint32_t* counterBufferPtr = nullptr;
        }mFalcorResources;
    };
}
