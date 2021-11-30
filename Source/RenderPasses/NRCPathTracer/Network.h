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
        float3 pos;     // the frequency encoding needs inputs to be mapped into [-1, 1]
        float roughness;// 
        float2 dir;     // the oneblob encoding needs inputs to be mapped in [0, 1]
        float2 normal;  // 
        float3 diffuse; // these parameters are passed through, since they preserve linearity    
        float _pad0;
        float3 specular;
        float _pad1;
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
        void reset();

        __host__ void inference(RadianceQuery* queries, cudaSurfaceObject_t output, uint32_t width, uint32_t height);
        __host__ void train(RadianceQuery* self_queries, uint32_t* self_query_counter,
            RadianceSample* training_samples, uint32_t* training_sample_counter, float& loss);

    private:
        uint32_t seed = 7272u;
    };
}

