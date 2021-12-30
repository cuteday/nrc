/** Path tracing pass.

    This file contains the entry points for all ray tracing programs.
    We import the path tracer utility functions defined in PathTracer.slang.

    The host sets the compile-time constants in StaticParams.slang.
    It also sets the following defines for optional I/O buffers:

    is_valid_<name> is 1 if buffer with this name is bound, 0 otherwise.
*/
import NRCPathTracer;
import Utils.Timing.GpuTimer;
import RenderPasses.Shared.PathTracer.LoadShadingData;

ParameterBlock<PathTracerData> gData;

// Outputs (optional)
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gOutputAlbedo;
RWTexture2D<uint> gOutputTime;

// Static configuration based on which buffers are bound.
#define is_valid(name) (is_valid_##name != 0)


/** ********************* Ray index 0: Scatter ray ************************ */

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData : SV_RayPayload)
{
}

[shader("anyhit")]
void scatterAnyHit(inout ScatterRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
#endif
}

[shader("closesthit")]
void scatterClosestHit(inout ScatterRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Store hit information. Note we don't access the materials here.
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    rayData.packedHitInfo = HitInfo(triangleHit).pack();
}

/************************** Ray index 1: Shadow ray ************************ */

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData : SV_RayPayload)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
#endif
}

/** ******************************** RayGen ******************************** */

/** This is the entry point for the path tracer.

    We generate N paths (= #spp) per pixel, which are traced into the scene.
    The path tracer is written as a for-loop over path segments, where each
    iteration traces a shadow ray for direct illumination and a scatter ray.

    The hit shader for the scatter ray currently generates ray parameters for
    the shadow ray to evaluate direct illumination and generates ray parameters
    for the next scatter ray, which are both returned the raygen shader to be
    traced. This is more efficient than tracing from the hit shader. The max
    recusion depth = 1.
*/
[shader("raygeneration")]
void rayGen()
{
    uint2 currentPixel = getCurrentPixelNRC(DispatchRaysIndex().xy);
    uint2 screenSize = gNRCScreenSize;

    logSetPixel(currentPixel);
    printSetPixel(currentPixel);
    logSetPixelNRC(currentPixel);
    NRCSetPixel(currentPixel);

    GpuTimer timer;
    if (is_valid(gOutputTime)) timer.start();

    float3 outColor = float3(0, 0, 0);
    float3 outAlbedo = float3(0, 0, 0);
    float outAlpha = 0.f;

    HitInfo hit;
    ShadingData sd;
    // loads the geometry info of current pixel (primary hit) into ShadingData sd and HitInfo hit
    if (loadShadingData(currentPixel, screenSize, gScene.camera, sd, hit))
    {
        // Pixel represents a primary hit. Compute its contribution.

        // Compute ray origin for new rays spawned from the G-buffer.
        const float3 rayOrigin = sd.computeNewRayOrigin();
        const NRCPathType pathType = getNRCPathType(currentPixel);

        // Loop over samples in pixel.
        [loop]
        for (uint sampleIdx = 0; sampleIdx < kSamplesPerPixel; sampleIdx++)
        {
            // Setup path data.
            PathData path = {};
            path.origin = rayOrigin;
            path.thp = float3(1.f);
            path.hit = hit;

            // Create sample generator.
            uint frameSeed = gData.params.useFixedSeed ? 0 : gData.params.frameCount;
            path.sg = SampleGenerator.create(currentPixel, frameSeed * kSamplesPerPixel + sampleIdx);

            // Advance the generator to the first available dimension.
            // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
            for (uint i = 0; i < gData.params.prngDimension; i++) sampleNext1D(path.sg);

            // TODO: Use (kRayFootprintMode != TexLODMode::Mip0) when slang is fixed.
            if (!(kRayFootprintMode == TexLODMode::Mip0))
            {
                // Create the ray footprint data for TexLOD.
                path.rayFootprint = RayFootprint.create(hit.getTriangleHit(), currentPixel, screenSize, rayOrigin, gScene.camera.getPosition(), sd.faceN, sd.N, gData.params.screenSpacePixelSpreadAngle, sd.linearRoughness, path.isSpecular());
            }

            // Trace the path.
            float3 L;
#ifdef NRC_ENABLE
            if (!gVisualizeMode || gIsTrainingPass) L = tracePathNRC(gData, sd, path);
            if (gVisualizeMode && !gIsTrainingPass) L = visualizeNRC(sd); 
#else
            L = tracePath(gData, sd, path);
#endif
            logPathLength(path.length);

            // Accumulate after clamping.
            // Note the comparison is written so that NaNs propagate (unless the compiler rewrites it).
            // TODO: Check the generated code that this is the case.
            outColor += gData.params.clampSamples && L > gData.params.clampThreshold ? gData.params.clampThreshold : L;
        }

        // We're done accumulating over all samples.
        const float invSpp = 1.f / kSamplesPerPixel;
        outColor *= invSpp;
        outAlbedo = sd.diffuse + sd.specular;
        outAlpha = 1.f;
    }
    else
    {
        // Background pixel.
        outColor = evalBackground(-sd.V);
        outAlbedo = outColor.rgb;
        outAlpha = kForceAlphaOne ? 1.f : 0.f;
#ifdef NRC_ENABLE
        gScreenQueryFactor[currentPixel] = float4(0.0);
        gScreenQueryBias[currentPixel] = float4(outColor, 1.0);
#endif
}

    // DEBUG
    assert(!any(isnan(outColor)));

    // Write outputs.
    // These are all optional so using compile-time checks to decide which ones to write.
    if (is_valid(gOutputColor)) gOutputColor[currentPixel] = float4(outColor, outAlpha);
    if (is_valid(gOutputAlbedo)) gOutputAlbedo[currentPixel] = float4(outAlbedo, 1);

    // Write time.
    if (is_valid(gOutputTime)) gOutputTime[currentPixel] = timer.getElapsed();
}
