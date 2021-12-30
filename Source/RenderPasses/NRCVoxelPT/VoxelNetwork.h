#pragma once
#include "json/json.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <memory>
#include <fstream>
#include "DataStructure.slang"

#ifndef NRC_VOXEL_MACRO
#define NRC_VOXEL_MACRO

#define voxel_flatten(index, voxel_size) (index.z + voxel_size.z*index.y + voxel_size.y*voxel_size.z*index.x)

#endif // !NRC_VOXEL_MACRO

namespace NRC {
    //namespace nlohmann { extern class json; }
    using nlohmann::json;

    class VoxelNetwork {

    public:
        using SharedPtr = std::shared_ptr<VoxelNetwork>;
        using WeakPtr = std::weak_ptr<VoxelNetwork>;
        using SharedConstPtr = std::shared_ptr<const VoxelNetwork>;

        // VoxelNetwork() = default;
        VoxelNetwork(json config);
        ~VoxelNetwork();

        void initializeNetwork(json net_config);
        void reset();
        float& learningRate() { return m_learning_rate; };

        __host__ void beginFrame(uint32_t* counterBufferDevice);
        __host__ void inference(RadianceQuery* queries, Falcor::uint2* pixels, cudaSurfaceObject_t output);
        __host__ void train(RadianceQuery* self_queries, RadianceSample* training_samples, float& loss);

    private:
        uint32_t seed = 7272u;
        float m_learning_rate = 1e-4f;
        
    };
}

