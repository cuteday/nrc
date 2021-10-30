#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.intersection;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Debug.PixelDebug;

import Experimental.Scene.Material.StandardMaterial;
import Experimental.Scene.Material.MaterialHelpers;
import Experimental.Scene.Lights.LightHelpers;
import Experimental.Scene.Lights.EnvMapLighting;

cbuffer RayTraceCB
{
    uint gFrameCount;
    uint gDirectSampleCount;
    float gMinT;
}

static struct Data
{
    // Materials
    StandardMaterial standardMaterial;
} gData;


Texture2D<float4> gWorldPos;
Texture2D<float4> gShadingNormals;
Texture2D<float4> gFaceNormals;
Texture2D<float4> gShadingTangent;
Texture2D<float4> gEnvMap;
// material buffers
Texture2D<float4> gDiffuseOpacity;
Texture2D<float4> gSpecRough;
Texture2D<float4> gEmissive;
Texture2D<float4> gMtlExtraParams;

RWTexture2D<float4> gColorOut;

struct ShadowRayData
{
    float visibility;
}

bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    RayDesc ray = { origin, gMinT, dir, distance };
    ShadowRayData rayData = { 1.0 };
    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff,
        0, /*hit index*/ rayTypeCount, /*intrinsic*/ 0, /*miss index*/
        ray, rayData);
    return rayData.visibility;
}

float3 evalDirectAnalytic(const ShadingData sd, float3 rayOrigin, inout SampleGenerator sg)
{
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0) return float3(0);

    // Pick one of the analytic light sources randomly with equal probability.
    const uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
    float invPdf = lightCount; // Light selection pdf = 1.0 / lightCount.

    // Sample local light source.
    AnalyticLightSample ls;
    bool valid = sampleLight(rayOrigin, gScene.getLight(lightIndex), sg, ls);

    // Reject sample if lower hemisphere.
    if (!valid || dot(ls.dir, sd.N) <= kMinCosTheta) return float3(0);

    // Test visibility by tracing a shadow ray.
    bool V = traceShadowRay(rayOrigin, ls.dir, ls.distance);
#ifdef SHADING_PBR
    return V * gData.standardMaterial.eval(sd, ls.dir) * ls.Li * invPdf;
#else
    return V * dot(sd.N, ls.dir) * ls.Li * invPdf;
#endif
}

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;         // pixel pos by default
    uint2 launchDim = DispatchRaysDimensions().xy;      // screen size

    float4 worldPos = gWorldPos[launchIndex];
    float3 shadingNormal = gShadingNormals[launchIndex].xyz;
    float3 faceNormal = gFaceNormals[launchIndex].xyz;
//    float4 tangent = gShadingTangent[launchIndex];
    float4 tangent = float4(perp_stark(shadingNormal), 1.f);
    float3 diffuse = gDiffuseOpacity[launchIndex].xyz;

    //const float3 primaryRayDir = getPrimaryRayDir(launchIndex, launchDim, gScene.camera);
    const float3 primaryRayDir = gScene.camera.computeRayPinhole(launchIndex, launchDim).dir;

    if (worldPos.w == 0.0f)
    {
        // evaluate skybox color
        float3 Le = gScene.envMap.eval(primaryRayDir);
        gColorOut[launchIndex] = float4(Le, 1.0);
        return;
    }

    float3 shadeColor = float3(0.0f);
    SampleGenerator sg = SampleGenerator.create(launchIndex, gFrameCount);

    // Helper routines in material helpers
    GeometryParams geoParams = prepareGeometryParams(worldPos.xyz,
            -primaryRayDir, shadingNormal, tangent, faceNormal);
    MaterialParams matParams;
    matParams.diffuseOpacity = gDiffuseOpacity[launchIndex];
    matParams.specularRoughness = gSpecRough[launchIndex];
    matParams.emissive = gEmissive[launchIndex];
    matParams.extraParams = gMtlExtraParams[launchIndex];
    
    ShadingData sd = prepareShadingData(geoParams, matParams);
#define SHADING_MONTE_CARLO
#ifdef SHADING_MONTE_CARLO
    for (int i = 0; i < gDirectSampleCount; i++)
    {
        shadeColor += evalDirectAnalytic(sd, worldPos.xyz, sg);
    }
    shadeColor /= gDirectSampleCount;
#else
    for(int i = 0; i < gScene.getLightCount(); i++)
    {
        AnalyticLightSample ls;
        LightData light = gScene.getLight(i);
        if (!sampleLight(worldPos.xyz, light, sg, ls))
            continue;
        float G = dot(shadingNormal, ls.dir) * traceShadowRay(worldPos.xyz, ls.dir, ls.distance);
        shadeColor += ls.Li * diffuse / M_PI * G / (ls.pdf > 0.f ? ls.pdf : 1.f);
    }
#endif
    gColorOut[launchIndex] = float4(shadeColor, 1.0f);
}

[shader("miss")]
void miss(inout ShadowRayData rayData)
{
}

[shader("anyhit")]
void anyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f))
            IgnoreHit();
}

[shader("closesthit")]
void closestHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    rayData.visibility = 0.0;
}
