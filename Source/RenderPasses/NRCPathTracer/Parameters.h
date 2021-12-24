#pragma once
#include "Utils/Math/Vector.h"
#include <cmath>

/*
    Parameters that don't need to change / can't be changed during runtime.
*/

#ifndef NRC_PARAMETERS
#define NRC_PARAMETERS

#define AUX_INPUTS          1
#define REFLECTANCE_FACT    1
#define LITE_SCREEN         0

namespace NRC {
    using Falcor::uint2;

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
        const unsigned int max_inference_query_size = resolution;
        const unsigned int max_training_query_size = 1 << 16;                   // ~57,600

        const uint2 trainingPathStrideRR = trainingPathStride * 4u;

        const unsigned int max_training_sample_size = resolution / trainingPathStride.x / trainingPathStride.y * 15;
        const uint32_t self_query_batch_size = resolution / trainingPathStride.x / trainingPathStride.y;     // ~ 57600

#if AUX_INPUTS
        const uint32_t input_dim = 14;         // pos dir normal roughness diffuse specular
#else
        const uint32_t input_dim = 5;           // pos, dir
#endif
        const uint32_t output_dim = 3;        // RGB
        //const uint32_t alignment = 16;        // input dim alignment
        const std::string config_path = "../RenderPasses/NRCPathTracer/Data/default_nrc_new.json";

    }
}

#endif
