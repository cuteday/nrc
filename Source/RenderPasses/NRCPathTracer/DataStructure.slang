#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_NRC

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

    struct RadianceInfer
    {
        RadianceQuery query;
        uint2 pixel;
        float2 _pad0;
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

    enum class NRCVisualizeMode {
        Result = 1,
        Radiance = 2,
        Factor = 3,
        Bias = 4,
        Reflectance = 5
    };

END_NAMESPACE_NRC
