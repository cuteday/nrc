#pragma once
#include "json/json.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <memory>
#include <fstream>

#include "DataStructure.slang"
#include "Utils/Math/Vector.h"

#ifndef NRC_VOXEL_MACRO
#define NRC_VOXEL_MACRO

#define voxel_flatten(index, voxel_size) (index.z + voxel_size.z*index.y + voxel_size.y*voxel_size.z*index.x)

#endif // !NRC_VOXEL_MACRO

namespace NRC {
    using Falcor::uint2;
    struct NRCResource {
        // cuda device pointers in unified memory space.
        NRC::RadianceQuery* inferenceQuery = nullptr;
        cudaSurfaceObject_t screenResult;       // write inferenced results here
        NRC::RadianceQuery* trainingQuery = nullptr;
        NRC::RadianceSample* trainingSample = nullptr;
        // voxel related
        uint32_t* trainingSampleCounter = nullptr;
        uint32_t* trainingQueryCounter = nullptr;
        uint32_t* inferenceQueryCounter = nullptr;
    };
}

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

        __host__ void initializeNetwork(json net_config);
        __host__ void registerResource(NRCResource resource);
        __host__ void reset();
        __host__ void flush() { cudaDeviceSynchronize(); }
        __host__ void debug();
        __host__ float& learningRate() { return m_learning_rate; };

        __host__ void prepareTraining();
        __host__ void prepareInference();
        __host__ void inference();
        __host__ void train(float& loss);

    private:
        uint32_t seed = 7272u;
        float m_learning_rate = 1e-4f;
        int m_network_width = 64;
        NRC::NRCResource mResource;
    };
}

