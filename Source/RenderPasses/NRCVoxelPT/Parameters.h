#pragma once
#include "Utils/Math/Vector.h"
#include <cmath>

/*
    Parameters that don't need to change / can't be changed during runtime.
*/

#ifndef NRC_PARAMETERS
#define NRC_PARAMETERS

#define NRC_VOXEL           1
#define AUX_INPUTS          1
#define REFLECTANCE_FACT    1
#define LITE_SCREEN         0

#define THRUST_INFERENCE_SORT 0

namespace NRC {
    using Falcor::uint2;
    using Falcor::uint3;

    namespace Parameters {

        // 4 steps each frame, with 16384 samples per batch
#if LITE_SCREEN
        const uint2 screen_size = uint2(1280, 720);
#else
        const uint2 screen_size = uint2(1920, 1080);
#endif
        const uint2 trainingPathStride = uint2(6, 6);
        const uint32_t batch_size = 1 << 14;
        const uint32_t n_train_batch = 4;

        const uint32_t resolution = screen_size.x * screen_size.y;    // is a multiple of 256
        const unsigned int max_inference_query_size = resolution + 1024;
        const unsigned int max_training_query_size = 1 << 16;                   // ~57,600

        const unsigned long long random_seed = 7272ULL; 

        const uint2 trainingPathStrideRR = trainingPathStride * 1u;

        const unsigned int max_training_sample_size = 1920 * 1080 * 15 / 36;
        const uint32_t self_query_batch_size = resolution / trainingPathStride.x / trainingPathStride.y;     // ~ 57600

        const int network_width = 32;
#if AUX_INPUTS
        const uint32_t input_dim = 14;         // pos dir normal roughness diffuse specular
#else
        const uint32_t input_dim = 5;           // pos, dir
#endif
        const uint32_t output_dim = 3;        // RGB
        //const uint32_t alignment = 16;        // input dim alignment
        const std::string config_path = "../RenderPasses/NRCVoxelPT/Data/default_nrc.json";

        // voxel related settings
        const uint32_t max_training_sample_voxel = 1 << 16;
        //const uint32_t max_inference_query_voxel = resolution;
        struct VoxelConfig {
            uint3 voxel_size = { 1, 1, 1 };
            unsigned int voxel_num = 1;
        };
        extern VoxelConfig voxel_param;
    }

    struct RadianceQueryCompact{
        float data[Parameters::input_dim];
    };
}

#endif
