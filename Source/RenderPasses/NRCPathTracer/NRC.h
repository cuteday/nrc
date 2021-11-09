#pragma once
#include <cuda_runtime.h>

#include <Falcor.h>

#include "Network.h"
#include "FalcorCUDA.h"

using Falcor::uint;
using Falcor::uint2;
using Falcor::uint3;
using Falcor::uint4;
using Falcor::float2;
using Falcor::float3;
using Falcor::float4;

namespace NRC {

    struct RadianceQuery {
        float3 pos;
        float2 dir;
    };

    struct RadianceRecord {
        float3 radiance;
    };

    class NRCInterface {

    public:
        using SharedPtr = std::shared_ptr<NRCInterface>;
        using WeakPtr = std::weak_ptr<NRCInterface>;
        using SharedConstPtr = std::shared_ptr<const NRCInterface>;

        NRCInterface();

        void trainFrame();

        void inferenceFrame();

    private:

        NRCNetwork::SharedPtr network = nullptr;
    };
}
