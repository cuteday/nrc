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
    bool gVisualizeMode;
    bool gIsTrainingPass;               // If we are in the training pass, dispatch training suffix via sparse sampling
    float gNRCAbsorptionProb;
    float gTerminateFootprintThres;     // hyperparam for ray termination heuristic
    uint2 gNRCScreenSize;               // width x height
    uint2 gNRCTrainingPathStride;       // uniform sparse sample paths for training
    uint2 gNRCTrainingPathOffset;       // random offset each frame, for the sparse sample
    uint2 gNRCTrainingPathStrideRR;     // training paths ending with russian roulette

    float3 gSceneAABBCenter;             
    float3 gSceneAABBExtent;
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
        if (pdf == 0) current = 0;      // sampled using a delta func.
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
{   // 32byte
    float3 pos;
    float roughness;
    float2 dir;
    float2 normal;
    float4 diffuse;
    float4 specular;
};

struct RadianceSample
{   // 64byte
    RadianceQuery query;
    // L_o (scattered radiance) = a * L_i + b
    int idx;    // which query it belongs to?
    float3 a;   // factor of scatter ray (bsdf sample)
    float3 b;   // the direct sample part
    float _pad0;
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
RWTexture2D<float4> gOutputResult;

// map and normalize the world coord to [-1, 1]
float3 normalizeCoord(float3 x)
{
    //return x;
    //return (x - gSceneAABBCenter) / gSceneAABBExtent + 1.0;
    return (x - gSceneAABBCenter) * 2 / gSceneAABBExtent; 
}

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

bool isTrainingPath()
{
    uint2 rel_coord = gNRCPixel - gNRCTrainingPathOffset;
    uint2 rel_index = rel_coord % gNRCTrainingPathStride;
    if (all(rel_index == 0))
        return true;
    return false;
}

bool isTrainingPathRR()
{
    uint2 rel_coord = gNRCPixel - gNRCTrainingPathOffset;
    uint2 rel_index = rel_coord % gNRCTrainingPathStrideRR;
    if (all(rel_index == 0))
        return true;
    return false;
}

RadianceQuery generateQuery(ShadingData sd)
{
    RadianceQuery query = { };
    query.pos = normalizeCoord(sd.posW);
    query.dir = world_to_latlong_map(sd.V);
    query.normal = world_to_latlong_map(sd.faceN);    // TODO: reverse the normal if the primitive is seen on the back-face side.
    query.diffuse = float4(sd.diffuse, 0);
    query.specular = float4(sd.specular, 0);
    query.roughness = sd.linearRoughness;
    return query;
}

PathVertexRecord generatePathVertexRecord(PathData path, ShadingData sd)
{
    RadianceQuery query = generateQuery(sd);
    PathVertexRecord record = { query, path.thp, path.L };
    return record;
}

void writeScreenInferenceQuery(ShadingData sd, float3 factor, float3 bias)
{
    RadianceQuery query = generateQuery(sd);
    uint2 frameDim = gNRCScreenSize;
    int index = frameDim.x * gNRCPixel.y + gNRCPixel.x;
    gScreenQueryFactor[gNRCPixel] = float4(factor, 0);
    gScreenQueryBias[gNRCPixel] = float4(bias, 1);
    gInferenceRadianceQuery[index] = query;
}

uint addTrainingQuery(ShadingData sd)
{
    uint idx = gTrainingRadianceQuery.IncrementCounter();
    RadianceQuery query = generateQuery(sd);
    gTrainingRadianceQuery[idx] = query;
    return idx;
}

void addTrainingSample(float3 a, float3 b, RadianceQuery query, int queryIndex)
{
    if (any(isnan(a)) || any(isnan(b)) || any(isinf(a)) || any(isinf(b)))
        // TODO: is isnan() in hlsl works properly?
        return;
    uint idx = gTrainingRadianceSample.IncrementCounter();
    RadianceSample sample = { };
    sample.query = query;
    sample.idx = queryIndex;
    sample.a = a;
    sample.b = b;
    gTrainingRadianceSample[idx] = sample;
}
