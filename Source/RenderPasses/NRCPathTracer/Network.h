#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <memory>
#include <fstream>
#ifdef __NVCC__
#define NRC_CALLABLE __host__ __device__
#else
#define NRC_CALLABLE
#endif

#include "DataStructure.slang"

namespace NRC {


    class NRCNetwork {

    public:
        using SharedPtr = std::shared_ptr<NRCNetwork>;
        using WeakPtr = std::weak_ptr<NRCNetwork>;
        using SharedConstPtr = std::shared_ptr<const NRCNetwork>;

        NRCNetwork();
        ~NRCNetwork();

        void initializeNetwork();
        void reset();

        __host__ void inference(RadianceQuery* queries, cudaSurfaceObject_t output, uint32_t width, uint32_t height);
        __host__ void train(RadianceQuery* self_queries, uint32_t* self_query_counter,
            RadianceSample* training_samples, uint32_t* training_sample_counter, float& loss);

    private:
        uint32_t seed = 7272u;
    };
}

