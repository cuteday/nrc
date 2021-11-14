#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <memory>
#include <fstream>




namespace NRC {
    struct RadianceQuery
    {
        float3 pos;
        float2 dir;
    };

    struct RadianceRecord
    {
        float3 radiance;
    };

    struct RadianceSample
    {
        // L_o (scattered radiance) = a * L_i + b
        float3 a;   // factor of scatter ray (bsdf sample)
        float3 b;   // the direct sample part
        int idx;    // which query it belongs to?
    };

    class NRCNetwork {

    public:
        using SharedPtr = std::shared_ptr<NRCNetwork>;
        using WeakPtr = std::weak_ptr<NRCNetwork>;
        using SharedConstPtr = std::shared_ptr<const NRCNetwork>;

        NRCNetwork();

    private:
        void initializeNetwork();
        void inference(RadianceQuery* queries, uint32_t n_elements);
    };
}
