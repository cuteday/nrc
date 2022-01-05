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

        void prepare();

        void trainFrame();

        void inferenceFrame();

        void printStats();

        void resetParameters();

        bool& enableTraining() { return mEnableTraining; }
        bool& enableInference() { return mEnableInference; }

        void registerNRCResources(Falcor::Buffer::SharedPtr pInferenceQueryBuffer,
            Falcor::Texture::SharedPtr pScreenResultTexture,
            Falcor::Buffer::SharedPtr pTrainingQueryBuffer,
            Falcor::Buffer::SharedPtr pTrainingSampleBuffer,
            Falcor::Buffer::SharedPtr pInferenceQueryCounter,
            Falcor::Buffer::SharedPtr pTrainingSampleCounter,
            Falcor::Buffer::SharedPtr pTrainingQueryCounter) {
            mParameters.screenSize = uint2(pScreenResultTexture->getWidth(), pScreenResultTexture->getHeight());
            mResource.screenResult = FalcorCUDA::mapTextureToSurfaceObject(pScreenResultTexture, cudaArrayColorAttachment);
            mResource.inferenceQuery = (NRC::RadianceQuery*)pInferenceQueryBuffer->getCUDADeviceAddress();
            
            mResource.trainingQuery = (NRC::RadianceQuery*)pTrainingQueryBuffer->getCUDADeviceAddress();
            mResource.trainingSample = (NRC::RadianceSample*)pTrainingSampleBuffer->getCUDADeviceAddress();

            mResource.inferenceQueryCounter = (uint32_t*)pInferenceQueryCounter->getCUDADeviceAddress();
            mResource.trainingSampleCounter = (uint32_t*)pTrainingSampleCounter->getCUDADeviceAddress();
            mResource.trainingQueryCounter = (uint32_t*)pTrainingQueryCounter->getCUDADeviceAddress();
            mNetwork->registerResource(mResource);
        }

        VoxelNetwork::SharedPtr mNetwork = nullptr;

    private:
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
        NRCResource mResource;

        bool mEnableTraining = true;
        bool mEnableInference = true;
    };
}
