import Scene.ShadingData;
import Utils.Math.MathHelpers;
import RenderPasses.Shared.PathTracer.PathData;

#ifndef NRC_MAX_TRAINING_BOUNCES
#define NRC_MAX_TRAINING_BOUNCES 5
#endif
#ifndef NRC_MAX_TRAINING_RR_BOUNCES
#define NRC_MAX_TRAINING_RR_BOUNCES 10
#endif
#ifndef NRC_MAX_INFERENCE_BOUNCES
#define NRC_MAX_INFERENCE_BOUNCES 5
#endif

cbuffer NRCDataCB
{   // the constant buffer is globally shared between shader files which included this file
    bool gNRCEnable;
    bool gIsTrainingPass;               // If we are in the training pass, dispatch training suffix via sparse sampling
    float gNRCAbsorptionProb;
    float gTerminateFootprintThres;     // hyperparam for ray termination heuristic
    uint2 gNRCScreenSize;               // width x height
    uint2 gNRCTrainingPathStride;       // uniform sparse sample paths for training
    uint2 gNRCTrainingPathOffset;       // random offset each frame, for the sparse sample
    uint2 gNRCTrainingPathStrideRR;     // training paths ending with russian roulette
};

enum NRCPathType
{
    InferencePath   = 0,
    TrainingPath    = 1,
    TrainingPathRR  = 2
};

struct NRCRayFootprint{
    float spread;

    NRCRayFootprint nextBounce(float3 u, float3 v, float pdf, float NdotV)
    { 
        NdotV = max(NdotV, 0.01);
        float current = distance(u, v) / pdf / NdotV;
        return { spread + current };
    }

    bool terminate()
    {
        return spread > gTerminateFootprintThres;
    }
};

static uint2 gNRCPixel;
void NRCSetPixel(uint2 pixel)
{
    gNRCPixel = pixel;
}

struct RadianceQuery
{
    float3 pos;
    float2 dir;
};

struct RadianceSample
{
    RadianceQuery query;
    // L_o (scattered radiance) = a * L_i + b
    float3 a;   // factor of scatter ray (bsdf sample)
    float3 b;   // the direct sample part
    int idx;    // which query it belongs to?
};

struct PathVertexRecord
{
    RadianceQuery query;
    float3 thp;     // current path throughput (not including BxDF, pdf of this vertex)
    float3 L;       // current path contribution excluding scattering radiance from this vertex
};

// radiance training sample at maximum
// for RR training data, no radiance info is needed
RWStructuredBuffer<RadianceQuery> gTrainingRadianceQuery;
RWStructuredBuffer<RadianceSample> gTrainingRadianceSample;
// for self training data, additional radiance info is needed

// radiance query at screen resolution
RWStructuredBuffer<RadianceQuery> gInferenceRadianceQuery;
RWTexture2D<float4> gScreenQueryFactor;
RWTexture2D<float4> gScreenQueryBias;

NRCPathType getNRCPathType(uint2 pixel)
{
    uint2 rel_coord = pixel - gNRCTrainingPathOffset;
    uint2 rel_index = rel_coord % gNRCTrainingPathStride;
    if (all(rel_coord % gNRCTrainingPathStrideRR == 0))
        return NRCPathType::TrainingPathRR;
    if (all(rel_coord % gNRCTrainingPathStride == 0))
        return NRCPathType::TrainingPath;
    return NRCPathType::InferencePath;
}

NRCPathType getNRCPathType()
{
    return getNRCPathType(gNRCPixel);
}

uint2 getCurrentPixelNRC(uint2 launchIndex)
{
    if (gNRCEnable && gIsTrainingPass)
        return gNRCTrainingPathOffset + launchIndex * gNRCTrainingPathStride;
    return launchIndex;
}

//bool isTrainingPath(uint2 pixel)
bool isTrainingPath()
{
    uint2 rel_coord = gNRCPixel - gNRCTrainingPathOffset;
    uint2 rel_index = rel_coord % gNRCTrainingPathStride;
    if (all(rel_index == 0))
        return true;
    return false;
}

//bool isTrainingPathRR(uint2 pixel)
bool isTrainingPathRR()
{
    uint2 rel_coord = gNRCPixel - gNRCTrainingPathOffset;
    uint2 rel_index = rel_coord % gNRCTrainingPathStrideRR;
    if (all(rel_index == 0))
        return true;
    return false;
}

PathVertexRecord generatePathVertexRecord(PathData path)
{
    RadianceQuery query = { path.origin, world_to_latlong_map(-path.dir) };
    PathVertexRecord record = { query, path.thp, path.L };
    return record;
}

void writeScreenInferenceQuery(float3 pos, float3 dir, float3 factor, float3 bias)
{
    RadianceQuery query = { pos, world_to_latlong_map(dir) };
    uint2 frameDim = gNRCScreenSize;
    int index = frameDim.x * gNRCPixel.y + gNRCPixel.x;
    gScreenQueryFactor[gNRCPixel] = float4(factor, 1.0);
    gScreenQueryBias[gNRCPixel] = float4(bias, 1.0);
    gInferenceRadianceQuery[index] = query;
}

int addTrainingQuery(float3 pos, float3 dir)
{
    int idx = gTrainingRadianceQuery.IncrementCounter();
    RadianceQuery query = { pos, world_to_latlong_map(dir) };
    gTrainingRadianceQuery[idx] = query;
    return idx;
}

void addTrainingSample(float3 a, float3 b, RadianceQuery query, int queryIndex)
{
    int idx = gTrainingRadianceSample.IncrementCounter();
    RadianceSample record = { query, a, b, queryIndex };
    gTrainingRadianceSample[idx] = record;
}

