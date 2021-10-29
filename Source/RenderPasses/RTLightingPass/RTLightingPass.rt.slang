#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.intersection;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;

import Experimental.Scene.Lights.LightHelpers;
import Experimental.Scene.Lights.EnvMapLighting;

cbuffer RayTraceCB
{
    uint gFrameCount;
    float gMinT;
}

Texture2D<float4> gWorldPos;
Texture2D<float4> gNormals;
Texture2D<float4> gDiffuseMtl;
Texture2D<float4> gEnvMap;

RWTexture2D<float4> gColorOut;

struct RayPayload
{
    float visibility;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    float4 worldPos = gWorldPos[launchIndex];
    float3 worldNormal = gNormals[launchIndex].xyz;
    float3 diffuse = gDiffuseMtl[launchIndex].xyz;

    const float3 viewDir = gScene.camera.computeRayPinhole(launchIndex, launchDim).dir;

    if (worldPos.w == 0.0f)
    {
        // evaluate skybox color
        float3 Le = gScene.envMap.eval(viewDir);
        gColorOut[launchIndex] = float4(Le, 1.0);
        return;
    }

    float3 shadeColor = float3(0.0f);

    for(int i = 0; i < gScene.getLightCount(); i++)
    {
        LightData light = gScene.getLight(i);
        AnalyticLightSample ls;
        evalLightApproximate(worldPos.xyz, light, ls);

        RayPayload rayData = { 1.0 };
        RayDesc ray = { worldPos.xyz, gMinT, ls.dir, ls.distance };
        // end accel structure searching at first hit unless anyhit shader called ignoreHit()
        uint flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

        TraceRay(gScene.rtAccel, flags, 0xff, 0, rayTypeCount, 0, ray, rayData);

        float G = dot(worldNormal, ls.dir) * rayData.visibility;
        //shadeColor += diffuse / M_PI * ls.Li / ls.pdf * G;
        shadeColor += ls.Li * diffuse / M_PI * G / (ls.pdf > 0.f ? ls.pdf : 1.f);
    }
    shadeColor /= gScene.getLightCount();
    gColorOut[launchIndex] = float4(shadeColor, 1.0f);
}

[shader("miss")]
void miss(inout RayPayload rayData)
{
}

[shader("anyhit")]
void anyHit(inout RayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f))
            IgnoreHit();
}

[shader("closesthit")]
void closestHit(inout RayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    rayData.visibility = 0.0;
}
