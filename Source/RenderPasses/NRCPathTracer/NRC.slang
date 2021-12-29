#include "Utils/Math/MathConstants.slangh"

import Scene.ShadingData;
import Utils.Math.MathHelpers;
import Utils.Debug.PixelDebug;
import RenderPasses.Shared.PathTracer.PathData;

import DataStructure;

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
    float gFootprintThresInference;     // hyperparam for ray termination heuristic
    float gFootprintThresSuffix;
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
    float a0;
    bool first_bounce;
    bool suffix;

    // the ray bounces from U to V.
    NRCRayFootprint nextBounce(float3 u, float3 v, float pdf, float NdotV)
    {
        float a0_ = this.a0, spread_ = this.spread;
        float duv = distance(u, v);
        NdotV = max(NdotV, 0.01);
        if (this.first_bounce) {
            a0_ = duv * duv / NdotV / (4 * M_PI);
        }
        else {
            float current = pdf != 0 ? sqrt(duv * duv / pdf / NdotV) : 0;
            // sampled using a delta func if pdf == 0.
            spread_ += current;
        }
        return { spread_, a0_, false, this.suffix };
    }

    bool terminate()
    {
        float thres = this.suffix ? gFootprintThresSuffix : gFootprintThresInference;
        return !first_bounce && this.spread > this.a0 * thres;
    }
};

static uint2 gNRCPixel;
void NRCSetPixel(uint2 pixel)
{
    gNRCPixel = pixel;
}

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
RWStructuredBuffer<uint2> gInferenceRadiancePixel;
RWTexture2D<float4> gScreenQueryFactor;
RWTexture2D<float4> gScreenQueryBias;
RWTexture2D<float4> gScreenQueryReflectance;
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
    float3 faceN = sd.frontFacing ? sd.faceN : -sd.faceN; 
    query.normal = world_to_latlong_map(faceN);    // reverse the normal if the primitive is seen on the back-face side.
    /*    query.diffuse = float4(sd.diffuse, 1);
        query.specular = float4(sd.specular, 1);*/
    query.diffuse = sd.diffuse;
    query.specular = sd.specular;
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
    gScreenQueryFactor[gNRCPixel] = float4(factor, 0);
    gScreenQueryBias[gNRCPixel] = float4(bias, 1);
    gScreenQueryReflectance[gNRCPixel] = float4(sd.specular + sd.diffuse, 1);
    RadianceQuery query = generateQuery(sd);
    //uint2 frameDim = gNRCScreenSize;
    //int index = frameDim.x * gNRCPixel.y + gNRCPixel.x;
    if (any(factor))
    {
        uint index = gInferenceRadianceQuery.IncrementCounter();
        gInferenceRadianceQuery[index] = query;
        gInferenceRadiancePixel[index] = gNRCPixel;
    }
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
        // TODO: does isnan() in hlsl works properly?
        return;
    uint idx = gTrainingRadianceSample.IncrementCounter();
    RadianceSample sample = { };
    sample.query = query;
    sample.idx = queryIndex;
    sample.a = a;
    sample.b = b;
    gTrainingRadianceSample[idx] = sample;
}

/*
 *  Debug helper functions
 */
void debugRayFootprint(const PathData path, const NRCRayFootprint footprint)
{
    print("At path depth: ", path.length);
    print("Path pdf: ", path.pdf);
    print("Ray spread: ", footprint.spread);
    print("Path intersection: ", path.origin);
    print("Path a0: ", footprint.a0);
}
