import Scene.Raytracing;
import Scene.intersection;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;

cbuffer RayTraceCB
{
    uint gFrameCount;       //
    uint gSampleCount;      //
    float gAoRadius;        //
    float gMinT;            //
}

Texture2D<float4> gNormals;
Texture2D<float4> gColorIn;
Texture2D<float4> gWorldPos;

RWTexture2D<float4> gColorOut;  // read-write textures
RWTexture2D<float4> gAoMap;

struct RayPayload
{
    float aoVal;
}

float3x3 tbnFromNormal(float3 normal, SampleGenerator sg)
{
    // create a random TBN with normal -> +z
    float3 randDir = sampleNext3D(sg);
    float3 tangent = normalize(randDir - normal * dot(normal, randDir));
    float3 bitangent = cross(normal, tangent);
    // note thet the constructor fills the matrix in a row-major order
    return float3x3(tangent, bitangent, normal);
}


[shader("raygeneration")]
void rayGen()
{
    // aware of these built-in functions!
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    float4 worldPos = gWorldPos[launchIndex];
    float3 worldNormal = gNormals[launchIndex].xyz;

    float aoVal = 1.0f;
    if (worldPos.w != 0.0) {
        // summon the cutest sample generator then do hemisphere sampling ! > <
        aoVal = 0.0f;

        SampleGenerator sg = SampleGenerator.create(launchIndex, gFrameCount);
        float3x3 tbn = tbnFromNormal(worldNormal, sg);

        for (int i = 0; i < gSampleCount; i++) {
            float2 u2f = sampleNext2D(sg);
            float pdf = 0.f;
            // sample direction in +z upper hemisphere
            float3 wi = sample_cosine_hemisphere_concentric(u2f, pdf);
            float3 dir = mul(wi, tbn);

            RayPayload rayData = { 0.0 };
            RayDesc ray = { worldPos.xyz, gMinT, dir, gAoRadius };

            uint flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

            TraceRay(gScene.rtAccel,    // passed in automatically.
                flags,                  // set FLAGS for the current ray!
                0xff                    /* instanceInclusionMask */,
                0 /* hit idx */,
                rayTypeCount,           // passed in automatically ?
                0 /* miss idx */,
                ray,                    // ray desc
                rayData);               // ray payload

            aoVal += rayData.aoVal;
        }
        aoVal /= gSampleCount;
       
    }
    gAoMap[launchIndex] = float4(float3(aoVal), 1);
    gColorOut[launchIndex] = gColorIn[launchIndex] * gAoMap[launchIndex];
}

[shader("miss")]
void miss(inout RayPayload rayData)
{
    rayData.aoVal = 1.0;
}

[shader("anyhit")]
void anyHit(inout RayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // a copy-paste routine
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
}

[shader("closesthit")]
void closestHit(inout RayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
}
