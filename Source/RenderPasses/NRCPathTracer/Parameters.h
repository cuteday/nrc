#pragma once
#include "Utils/Math/Vector.h"

/*
    Parameters that don't need to change / can't be changed during runtime.
*/

#ifndef NRC_PARAMETERS
#define NRC_PARAMETERS

#define AUX_INPUTS          1
#define REFLECTANCE_FACT    1

//extern class uint2;

namespace NRC {
    using Falcor::uint2;

    namespace Parameters {

        const unsigned int max_inference_query_size = 1920 * 1080;
        const unsigned int max_training_query_size = 1 << 16;                   // ~57,600
        const unsigned int max_training_sample_size = 1920 * 1080 / 36 * 15;
        const uint2 trainingPathStride = uint2( 6, 6 );
        const uint2 trainingPathStrideRR = uint2( 24, 24 );
    }
}

#endif
