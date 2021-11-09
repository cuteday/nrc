__exported import PathTracer;

cbuffer NRCDataCB
{
    uint2 gNRCScreenSize;           // width x height
    uint2 gNRCTrainingPathStride;      // uniform sparse sample paths for training
    uint2 gNRCTrainingPathOffset;      // random offset each frame, for the sparse sample
};

struct RadianceQuery
{
    float3 pos;
    float2 dir;
};

struct RadianceRecord
{
    float3 radiance;
};

struct RadianceInfo
{
    // L_o (scattered radiance) = a * L_i + b
    float3 a;   // factor of scatter ray (bsdf sample)
    float3 b;   // the direct sample part
    int idx;    // which query it blongs to?
};

// radiance training sample at maximum 
// for RR training data, no radiance info is needed
RWStructuredBuffer<RadianceQuery> gTrainingRadianceQuery;
RWStructuredBuffer<RadianceRecord> gTrainingRadianceRecord;
// for self training data, additional radiance info is needed

// radiance query at screen resolution
RWStructuredBuffer<RadianceQuery> gInferenceRadianceQuery;
RWTexture2D<float4> gScreenQueryFactor;
RWTexture2D<float4> gScreenQueryBias;

bool isTrainingPath(uint2 pixel)
{
    uint2 rel_coord = pixel - gNRCTrainingPathOffset;
    uint2 rel_index = rel_coord % gNRCTrainingPathStride;
    if (all(rel_index == 0))
        return true;
    return false;
}

void writeScreenQuery(uint2 pixel, float3 pos, float2 dir, float3 factor, float3 bias)
{
    RadianceQuery query = { pos, dir };
    uint2 frameDim = gNRCScreenSize;
    int index = frameDim.x * pixel.y + pixel.x;
    gScreenQueryFactor[index] = float4(factor, 1.0);
    gScreenQueryBias[index] = float4(bias, 1.0);
    gInferenceRadianceQuery[index].pos = pos;
    gInferenceRadianceQuery[index].dir = dir;
}
