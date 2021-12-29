#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <memory>
#include <fstream>

#include "DataStructure.slang"

namespace NRC {



    class VoxelNetwork {

    public:
        using SharedPtr = std::shared_ptr<VoxelNetwork>;
        using WeakPtr = std::weak_ptr<VoxelNetwork>;
        using SharedConstPtr = std::shared_ptr<const VoxelNetwork>;

        VoxelNetwork();
        ~VoxelNetwork();

        void initializeNetwork();
        void reset();
        float& learningRate() { return learning_rate; };

        __host__ void beginFrame(uint32_t* counterBufferDevice);
        __host__ void inference(RadianceQuery* queries, Falcor::uint2* pixels, cudaSurfaceObject_t output);
        __host__ void inference(RadianceQuery* queries, cudaSurfaceObject_t output, uint32_t width, uint32_t height);
        __host__ void train(RadianceQuery* self_queries, uint32_t* self_query_counter,
            RadianceSample* training_samples, uint32_t* training_sample_counter, float& loss);

    private:
        uint32_t seed = 7272u;
        float learning_rate = 1e-4f;
        
    };
}

