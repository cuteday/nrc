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

namespace NRC {
    struct RadianceQuery
    {
        float3 pos;
        float _pad0;
        float2 dir;
        float2 _pad1;
    };

    struct RadianceSample
    {
        RadianceQuery query;
        // L_o (scattered radiance) = a * L_i + b
        int idx;    // which query it belongs to?
        float3 a;   // factor of scatter ray (bsdf sample)
        float3 b;   // the direct sample part
        float _pad0;
    };

    class NRCNetwork {

    public:
        using SharedPtr = std::shared_ptr<NRCNetwork>;
        using WeakPtr = std::weak_ptr<NRCNetwork>;
        using SharedConstPtr = std::shared_ptr<const NRCNetwork>;

        NRCNetwork();
        ~NRCNetwork();

        void initializeNetwork();
        __host__ void inference(RadianceQuery* queries, cudaSurfaceObject_t output, unsigned int width, unsigned int height);
        __host__ void train(RadianceQuery* self_queries, uint32_t* self_query_counter,
            RadianceSample* training_samples, uint32_t* training_sample_counter, float& loss);
    };
}

