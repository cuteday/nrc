/** Path tracer core functions.
    These are called from the raygen programs.

    To use it import PathTracer.slang and instantiate ParameterBlock<PathTracerData>,
    which is passed in as the first argument to all functions that need it.
    Any resources placed in PathTracerData are bound once and shared between all shaders/instances.
*/

#include "Utils/Math/MathConstants.slangh"

__exported import Scene.Scene;
__exported import Scene.Raytracing;
__exported import Scene.HitInfo;
__exported import Experimental.Scene.Material.StandardMaterial;
__exported import RenderPasses.Shared.PathTracer.PixelStats;
__exported import RenderPasses.Shared.PathTracer.PathTracerHelpers;
__exported import RenderPasses.Shared.PathTracer.InteriorListHelpers;
__exported import RenderPasses.Shared.PathTracer.RayFootprint;
__exported import Utils.Debug.PixelDebug;

__exported import NRC;
__exported import Debug.NRCPixelStats;

static const bool kForceOpaque = !kUseAlphaTest;

/** Shared path tracer data.
    The ray tracing program instantiates this and passes it to all functions.
*/
struct PathTracerData
{
    PathTracerParams        params;             ///< PathTracer shared parameters.

    // Samplers
    EnvMapSampler           envMapSampler;      ///< Environment map sampler.
    EmissiveLightSampler    emissiveSampler;    ///< Emissive light sampler.

    // Materials
    StandardMaterial        standardMaterial;   ///< Instance of Falcor's standard surface material.
};

struct ShadowRayData
{   // shadow ray payload
    bool visible;
};

struct ScatterRayData
{   // scatter ray payload
    PackedHitInfo packedHitInfo;        ///< Packed HitInfo data, or kInvalidIndex in the first component if ray missed.
};

/** Traces a shadow ray towards a light source.
    \param[in] origin Ray origin for the shadow ray.
    \param[in] dir Direction from ray origin towards the light source (normalized).
    \param[in] distance Distance to the light source.
    \param[in] valid True if ray should be traced, false for dummy rays.
    \return True if light is visible, false otherwise.
*/
bool traceShadowRay(float3 origin, float3 dir, float distance, bool valid = true)
{
    // Setup ray descriptor.
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = valid ? distance : 0.f; // Set tmax = 0.0 for dummy rays.

    ShadowRayData rayData = { false }; // Set to true by miss shader if ray is not terminated before
    uint rayFlags = RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    if (kForceOpaque) rayFlags |= RAY_FLAG_FORCE_OPAQUE;
    TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeShadow /* hitIdx */, rayTypeCount, kRayTypeShadow /* missIdx */, ray, rayData);

    // We let TraceRay() execute even when !valid in order to run the miss shader.
    // This is faster on current drivers thanks to reduced divergence.
    if (!valid) return false;

    // Note we're not counting stats for dummy rays. TODO: Make a separate bin in the logger.
    logTraceRay(PixelStatsRayType::Shadow);

    return rayData.visible;
}

/** Traces a scatter ray.
    \param[in] origin Ray origin for the shadow ray.
    \param[in] dir Direction from ray origin towards the light source (normalized).
    \param[in,out] interiorList Interior list for handling nested dielectrics.
    \param[out] hit Hit information. The type is set to 'HitType::None' upon miss.
    \return False if path was terminated, true otherwise.
*/
bool traceScatterRay(float3 origin, float3 dir, inout InteriorList interiorList, out HitInfo hit)
{
    hit = {};

    // Setup ray based on params passed via payload.
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    ScatterRayData rayData = {};
    uint rejectedHits = 0;

    // For nested dielectrics, we potentially have to trace additional rays after false intersections.
    while (true)
    {
        rayData.packedHitInfo.x = 0; // Clear hit

        uint rayFlags = RAY_FLAG_NONE; // TODO: Set cull mode from the app
        if (kForceOpaque) rayFlags |= RAY_FLAG_FORCE_OPAQUE;
        TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatter /* hitIdx */, rayTypeCount, kRayTypeScatter /* missIdx */, ray, rayData);

        logTraceRay(PixelStatsRayType::ClosestHit);

        // Check for false intersections.
        if (kUseNestedDielectrics && rayData.packedHitInfo.x != 0)
        {
            const TriangleHit triangleHit = HitInfo(rayData.packedHitInfo).getTriangleHit();
            uint materialID = gScene.getMaterialID(triangleHit.instanceID);
            uint nestedPriority = gScene.materials[materialID].getNestedPriority();
            if (!interiorList.isTrueIntersection(nestedPriority))
            {
                if (rejectedHits < kMaxRejectedHits)
                {
                    rejectedHits++;
                    VertexData v = gScene.getVertexData(triangleHit);
                    bool frontFacing = dot(-ray.Direction, v.faceNormalW) >= 0.f;
                    interiorList.handleIntersection(materialID, nestedPriority, frontFacing);
                    ray.Origin = computeRayOrigin(v.posW, frontFacing ? -v.faceNormalW : v.faceNormalW);
                    continue;
                }
                else
                {
                    // Terminate path.
                    return false;
                }
            }
        }
        break;
    }

    if (rayData.packedHitInfo.x != 0)
    {
        hit = HitInfo(rayData.packedHitInfo);
    }

    return true;
}

// Need to be kept alive until next scatter event for ray footprint to be updated after the ray bounces.
static StaticVertexData triangleVertices[3];
static VertexData v;

/** Handle scatter ray hit.
    \param[in] pt PathTracer data.
    \param[in] sd Shading data.
    \param[in,out] path Path data.
*/
void handleHit(const PathTracerData pt, inout ShadingData sd, inout PathData path)
{
    logPathVertex();

    // We only support triangle hits.
    const TriangleHit triangleHit = path.hit.getTriangleHit();

    // Get vertex data for current hit point. VertexData and triangleVertices kept alive out of handleHit() to be used by rayFootprint.bounceOnSurface() if needed.
    v = gScene.getVertexData(triangleHit, triangleVertices);

    // Update the ray footprint with current hit surface.
    path.rayFootprint.hitSurface(v, triangleVertices, triangleHit, path.origin, path.dir);

    // Evaluate Falcor's material parameters at the hit point using the current ray footprint mode and doing texLOD.
    sd = prepareShadingData(v, path.rayFootprint, triangleVertices, triangleHit, path.origin, path.dir);

    if (kAdjustShadingNormals) adjustShadingNormal(sd, v);

    if (kUseNestedDielectrics)
    {
        // Compute relative index of refraction at interface.
        sd.eta = computeRelativeIoR(path.interiorList, sd.materialID, sd.frontFacing);

        // Handle volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint interiorMaterialID = path.interiorList.getTopMaterialID();
            float3 sigmaA = gScene.materials[interiorMaterialID].volumeAbsorption;
            float distance = length(path.origin - v.posW);
            path.thp *= exp(-sigmaA * distance);
        }
    }

    // Disable specular lobes if caustics are disabled and path already contains non-specular vertex.
    if (kDisableCaustics && path.nonSpecularBounces > 0 && sd.linearRoughness <= pt.params.specularRoughnessThreshold)
    {
        sd.setActiveLobes((uint)LobeType::Diffuse);
    }

    // Check if the scatter event is samplable by the light sampling technique. (no delta & no transmission)
    const bool isLightSamplable = path.isLightSamplable();

    // Determine if we need to compute the emissive based on the current configuration.
    // It's only needed if emissive is enabled, and its full contribution hasn't been sampled elsewhere.
    const bool computeEmissive = kUseEmissiveLights && (kUseLightsInDielectricVolumes || !path.isInsideVolume()) && (!kUseNEE || kUseMIS || !isLightSamplable);

    if (computeEmissive && any(sd.emissive > 0.f))
    {
        float misWeight = 1.f;
        if (kUseNEE && kUseMIS && isLightSamplable)
        {
            // If emissive light sampling and MIS is enabled, we've already sampled emissive lights using NEE.
            // We need to evaluate the MIS weight here to account for the remaining contribution.
            // Note that MIS is only applied for hits on emissive triangles (other emissive geometry is not supported).

            // Prepare hit point struct with data needed for emissive light PDF evaluation.
            TriangleLightHit hit;
            hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
            hit.posW = sd.posW;
            hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;

            // Evaluate PDF at the hit, had it been generated with light sampling.
            float lightPdf = pt.emissiveSampler.evalPdf(path.origin, path.normal, true, hit) * getEmissiveLightSelectionPdf();

            // Compute MIS weight by combining this with BRDF sampling.
            // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
            misWeight = evalMIS(pt.params, 1, path.pdf, kLightSamplesPerVertex, lightPdf);
        }

        // Accumulate emitted radiance weighted by path throughput and MIS weight.
        path.L += path.thp * sd.emissive * misWeight;
    }

    path.length++;

    // Compute the ray origin for the next path segment.
    // The origin is offset slightly from the hit to avoid self-intersections.
    path.origin = sd.computeNewRayOrigin();
}

/** Handle scatter ray miss.
    \param[in] pt PathTracer data.
    \param[in,out] path Path data.
*/
void handleMiss(const PathTracerData pt, inout PathData path)
{
    // Check if the scatter event is samplable by the light sampling technique.
    const bool isLightSamplable = path.isLightSamplable();

    // If we have an environment, add it's weighted contribution here.
    if (kUseEnvLight && (kUseLightsInDielectricVolumes || !path.isInsideVolume()) && (!kUseNEE || kUseMIS || !isLightSamplable))
    {
        logPathVertex();

        float misWeight = 1.f;
        if (kUseNEE && kUseMIS && isLightSamplable)
        {
            // We came here through BRDF sampling. The other sampling strategy is
            // env map sampling. Evaluate it's probability for the current ray dir.
            float lightPdf = pt.envMapSampler.evalPdf(path.dir) * getEnvLightSelectionPdf();

            // Compute MIS weighted contribution from the environment map.
            misWeight = evalMIS(pt.params, 1, path.pdf, kLightSamplesPerVertex, lightPdf);
        }

        float3 Le = pt.envMapSampler.eval(path.dir);

        // Accumulate emitted radiance weighted by path throughput.
        path.L += path.thp * Le * misWeight;
    }
}

void evalDirect(const PathTracerData pt, ShadingData sd, inout PathData path)
{
    for (uint i = 0; i < kLightSamplesPerVertex; ++i)
    {
        ShadowRay shadowRay = {};
        bool valid = generateShadowRay(pt.params, pt.envMapSampler, pt.emissiveSampler, sd, pt.standardMaterial, i, path, path.sg, shadowRay);
        bool visible = traceShadowRay(path.origin, shadowRay.rayParams.xyz, shadowRay.rayParams.w, valid);
        path.L += visible ? shadowRay.Lr : float3(0.f);
    }
}

float3 tracePath(const PathTracerData pt, ShadingData sd, inout PathData path)
{
    logPathVertex();

    // Always output directly emitted light from the primary hit.
    path.L += sd.emissive;

    // We always add one bounce for handling surfaces that cannot be handled using light sampling.
    uint kIterationCount = kMaxBounces + 1;

    [loop]
    for (uint depth = 0; depth < kIterationCount; depth++)
    {
        // Determine if BSDF supports sampling with next-event estimation, The following line generates bad code:
        // bool supportsNEE = (getBSDFLobes(sd) & (uint)LobeType::NonDeltaReflection) != 0;
        uint lobes = pt.standardMaterial.getBSDFLobes(sd);
        bool supportsNEE = (lobes & (uint)LobeType::DiffuseReflection) != 0 || (lobes & (uint)LobeType::SpecularReflection) != 0;

        if (kUseNEE && supportsNEE && (kUseLightsInDielectricVolumes || !path.isInsideVolume()))
        {
            // NEE samples the emissive triangles | light sources | environment map, adds its contribution to path.L 
            evalDirect(pt, sd, path);
        }

        // Russian roulette to stochastically terminate the path.
        if (kUseRussianRoulette)
        {
            float u = sampleNext1D(path.sg);
            if (u < pt.params.probabilityAbsorption) break;
            path.thp /= (1.f - pt.params.probabilityAbsorption);
        }

        // Save incoming ray direction for ray footprint.
        const float3 rayDirIn = path.dir;

        // Generate next path segment. (Do BSDF importance sampling and update shading data & path data)
        // Here, the PathData scattering direction, throughput, and BxDF sampling pdf are updated.
        if (!generateScatterRay(pt.params, sd, pt.standardMaterial, path, path.sg)) break;

        // Scatter the ray footprint out of the surface. Primary bounce is handled at footprint creation time.
        if (depth > 0)
        {
            path.rayFootprint.bounceOnSurface(v, triangleVertices, path.hit.getTriangleHit(), sd, v.normalW, rayDirIn, path.dir, path.isSpecular(), path.isTransmission());
        }

        // Update ray origin on transmissive events.
        if (path.isTransmission()) path.origin = path.transmissionOrigin;

        const bool isLastVertex = path.length == kMaxBounces || path.nonSpecularBounces > kMaxNonSpecularBounces;

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (isLastVertex && (kUseNEE && !kUseMIS && path.isLightSamplable())) break;

        // Terminate caustics paths.
        if (kDisableCaustics && path.isSpecular() && path.nonSpecularBounces > 0) break;

        // Trace scatter ray, The path will either be directly terminated (for nested dielectrics) or a hit/miss is reported.
        if (!traceScatterRay(path.origin, path.dir, path.interiorList, path.hit)) break;

        const bool validHit = path.hit.isValid();

        if (validHit)
        {   // Handle current intersection, update path data & shading data for next scatter. Here the emissive radiance is added to path contribution.
            handleHit(pt, sd, path);
        }
        else
        {   // add MIS weighted contribution from environment map.
            handleMiss(pt, path);
        }

        if (!validHit) break;
        if (isLastVertex) break;
    }
    return path.L;
}

// records on vertices of a path, for tracing back scattering radiance of each vertex.
static PathVertexRecord pathRecords[NRC_MAX_TRAINING_RR_BOUNCES + NRC_MAX_INFERENCE_BOUNCES + NRC_MAX_TRAINING_BOUNCES];
float3 tracePathNRC(const PathTracerData pt, ShadingData sd, inout PathData path)
{
    // we'll skip paths which contain a training suffix at the inference pass.
    // just a workaround, the time cost of invoking this shader still exists.

    logPathVertex();
    NRCPathType pathType = getNRCPathType();
    path.L += sd.emissive;

    const uint kInferencePathLength = NRC_MAX_TRAINING_BOUNCES;
    bool normalTerminate = true;
    NRCRayFootprint inferenceRayFootprint = { 0.0 };

    /* inference loop! */
    [loop]
    for (uint depth = 0; depth < kInferencePathLength; depth++)
    {
        {   // do a training sample snapshot here
            pathRecords[path.length] = generatePathVertexRecord(path);
            // calculate and update ray footprint for termination heuristic
            inferenceRayFootprint.nextBounce(path.origin, sd.posW, path.pdf, sd.NdotV);
        }

        uint lobes = pt.standardMaterial.getBSDFLobes(sd);
        bool supportsNEE = (lobes & (uint)LobeType::DiffuseReflection) != 0 || (lobes & (uint)LobeType::SpecularReflection) != 0;

        if (kUseNEE && supportsNEE && (kUseLightsInDielectricVolumes || !path.isInsideVolume()))
        {
            // NEE samples the emissive triangles | light sources | environment map, adds its contribution to path.L 
            evalDirect(pt, sd, path);
        }

        // Save incoming ray direction for ray footprint.
        const float3 rayDirIn = path.dir;

        // Generate next path segment. (Do BSDF importance sampling and update shading data & path data)
        // Here, the PathData scattering direction, throughput, and BxDF sampling pdf are updated.
        if (!generateScatterRay(pt.params, sd, pt.standardMaterial, path, path.sg)) { normalTerminate = false; break; }

        // Scatter the ray footprint out of the surface. Primary bounce is handled at footprint creation time.
        if (depth > 0)
        {
            path.rayFootprint.bounceOnSurface(v, triangleVertices, path.hit.getTriangleHit(), sd, v.normalW, rayDirIn, path.dir, path.isSpecular(), path.isTransmission());
        }

        // Update ray origin on transmissive events.
        if (path.isTransmission()) path.origin = path.transmissionOrigin;

        const bool isQueryVertex = inferenceRayFootprint.terminate() /*ray footprint heuristic*/
            || path.length == NRC_MAX_INFERENCE_BOUNCES
            || path.nonSpecularBounces > kMaxNonSpecularBounces;

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (isQueryVertex && (kUseNEE && !kUseMIS && path.isLightSamplable())) { normalTerminate = false; break; }
        if (kDisableCaustics && path.isSpecular() && path.nonSpecularBounces > 0) { normalTerminate = false; break; }
        if (!traceScatterRay(path.origin, path.dir, path.interiorList, path.hit)) { normalTerminate = false; break; }

        const bool validHit = path.hit.isValid();

        if (validHit)
        {   // Handle current intersection, update path data & shading data for next scatter. Here the emissive radiance is added to path contribution.
            // also the length of the path is increased here
            handleHit(pt, sd, path);
        }
        else
        {   // add MIS weighted contribution from environment map.
            normalTerminate = false;
            handleMiss(pt, path);
        }

        if (isQueryVertex) break;
        if (!validHit) break;  
    }

    { 
        // we use these infomation to query the nrc, note that self emission is excluded from the prediction
        // the origin of the next bounce is updated in HandleHit
        // the direction is generated at the previous GenerateScatterRay call
        float3 scatterDir = -path.dir;
        float3 scatterOrigin = path.origin;
        float3 throughput = normalTerminate ? path.thp : float3(0.0);       // the path throughput is updated in generate scatter ray.
        float3 bias = path.L;               // the L already contains the self emission at the hit vertex
        writeScreenInferenceQuery(scatterOrigin, scatterDir, throughput, bias);
    }

    const float3 queryRadiance = path.L;
    const uint inferencePathLength = path.length;
    logInferencePathLength(inferencePathLength);
    if (!normalTerminate) return queryRadiance;   // we don't need to query the NRC...
    /* Path enlongation! (for RR or terminate heuristic) */
    if (!gIsTrainingPass) return queryRadiance;
    // save outputs at the query vertex

    // basically, if we needs to query the NRC, we use this flag and adds a query to the structured buffer.
    NRCRayFootprint suffixRayFootprint = { 0.0 };
    bool needsQuery = pathType == NRCPathType::TrainingPath;
    const uint kTrainingSuffixLength = pathType == NRCPathType::TrainingPathRR ?
        NRC_MAX_TRAINING_RR_BOUNCES : NRC_MAX_INFERENCE_BOUNCES;
    [loop]
    for(uint depth = 0; depth < kTrainingSuffixLength; depth++)
    {
        {   // do a training sample snapshot here
            pathRecords[path.length] = generatePathVertexRecord(path);
            suffixRayFootprint.nextBounce(path.origin, sd.posW, path.pdf, sd.NdotV);
        }
        uint lobes = pt.standardMaterial.getBSDFLobes(sd);
        bool supportsNEE = (lobes & (uint)LobeType::DiffuseReflection) != 0 || (lobes & (uint)LobeType::SpecularReflection) != 0;

        if (kUseNEE && supportsNEE && (kUseLightsInDielectricVolumes || !path.isInsideVolume()))
        {
            evalDirect(pt, sd, path);
        }

        if (pathType == NRCPathType::TrainingPathRR)
        {
            float u = sampleNext1D(path.sg);
            if (u < pt.params.probabilityAbsorption) break;
            path.thp /= (1.f - pt.params.probabilityAbsorption);
        }

        // Save incoming ray direction for ray footprint.
        const float3 rayDirIn = path.dir;

        if (!generateScatterRay(pt.params, sd, pt.standardMaterial, path, path.sg)) { needsQuery = false; break; };
        if (depth > 0)
        {
            path.rayFootprint.bounceOnSurface(v, triangleVertices, path.hit.getTriangleHit(), sd, v.normalW, rayDirIn, path.dir, path.isSpecular(), path.isTransmission());
        }
        
        if (path.isTransmission()) path.origin = path.transmissionOrigin;

        const bool isLastVertex = (needsQuery && suffixRayFootprint.terminate())
            || path.length == NRC_MAX_INFERENCE_BOUNCES
            || path.nonSpecularBounces > kMaxNonSpecularBounces;

        if (isLastVertex && (kUseNEE && !kUseMIS && path.isLightSamplable())) { needsQuery = false; break; }
        if (kDisableCaustics && path.isSpecular() && path.nonSpecularBounces > 0) { needsQuery = false; break; }
        if (!traceScatterRay(path.origin, path.dir, path.interiorList, path.hit)) { needsQuery = false; break; }

        const bool validHit = path.hit.isValid();

        if (validHit)
        {   // Handle current intersection, update path data & shading data for next scatter. Here the emissive radiance is added to path contribution.
            handleHit(pt, sd, path);
        }
        else
        {   // add MIS weighted contribution from environment map.
            handleMiss(pt, path);
            needsQuery = false;
        }

        if (!validHit) break;
    }

    logSuffixPathLength(path.length - inferencePathLength);
    int queryIndex = needsQuery ? addTrainingQuery(path.origin, -path.dir) : -1;

    for(uint i = 0; i < path.length; i++) {
        // from 0 to length-1 is indeed the best choice, for either [end with RR] or [missed] or [max bounces limit reached]
        // however, when [missed] the length-th record could actually be used.
        float3 sampleL = (path.L - pathRecords[i].L) / pathRecords[i].thp;
        float3 sampleFactor = path.thp / pathRecords[i].thp;
        addTrainingSample(sampleFactor, sampleL, pathRecords[i].query, queryIndex);   
    }

    return queryRadiance;
}
