/** Path tracer core functions.
    These are called from the raygen programs.

    To use it import PathTracer.slang and instantiate ParameterBlock<PathTracerData>,
    which is passed in as the first argument to all functions that need it.
    Any resources placed in PathTracerData are bound once and shared between all shaders/instances.
*/

#include "Utils/Math/MathConstants.slangh"

__exported import PathTracer;
__exported import NRC;
__exported import Debug.NRCPixelStats;


// records on vertices of a path, for tracing back scattering radiance of each vertex.
static PathVertexRecord pathRecords[NRC_MAX_TRAINING_RR_BOUNCES + NRC_MAX_INFERENCE_BOUNCES + NRC_MAX_TRAINING_BOUNCES];
float3 tracePathNRC(const PathTracerData pt, ShadingData sd, inout PathData path)
{
    // we'll skip paths which contain a training suffix at the inference pass.
    // just a workaround, the time cost of invoking this shader still exists.

    logPathVertex();
    NRCPathType pathType = getNRCPathType();
    path.L += sd.emissive;

    const uint kMaxInferenceIter = NRC_MAX_INFERENCE_BOUNCES + 1;
    bool normalTerminate = true;
    bool lastScatterMissed = false;
    NRCRayFootprint inferenceRayFootprint = { 0, 0, true, false };

    /* inference loop! */
    [loop]
    for (uint depth = 0; depth < kMaxInferenceIter; depth++)
    {
        {   // do a training sample snapshot here
            pathRecords[path.length] = generatePathVertexRecord(path, sd);
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
            || path.length == kMaxInferenceIter - 1
            || path.nonSpecularBounces > kMaxNonSpecularBounces;

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (isQueryVertex && (kUseNEE && !kUseMIS && path.isLightSamplable())) { normalTerminate = false; break; }
        if (kDisableCaustics && path.isSpecular() && path.nonSpecularBounces > 0) { normalTerminate = false; break; }
        if (!traceScatterRay(path.origin, path.dir, path.interiorList, path.hit)) { normalTerminate = false; break; }

        const bool validHit = path.hit.isValid();

        if (validHit)
        {   // Handle current intersection, update path data & shading data for next scatter. Here the emissive radiance is added to path contribution.
            // also the length of the path is increased here
            // [NOTE] that the path origin and the sd.posW is simultaneously updated here, but they slightly differences since the hit pos is slightly offseted on the surface.
            float3 prevPos = sd.posW;   // save the previous ray origin before it updates.
            handleHit(pt, sd, path);
            // update ray footprint information
            inferenceRayFootprint = inferenceRayFootprint.nextBounce(prevPos, sd.posW, path.pdf, sd.NdotV);
            debugRayFootprint(path, inferenceRayFootprint);
        }
        else
        {   // add MIS weighted contribution from environment map.
            lastScatterMissed = true;
            handleMiss(pt, path);
        }

        if (isQueryVertex || inferenceRayFootprint.terminate()) break;
        if (!validHit) break;  
    }

    if (!gIsTrainingPass && !gVisualizeMode)
    { 
        // we use these infomation to query the NRC, note that self emission is excluded from the prediction
        // the origin of the next bounce is updated in HandleHit
        // the direction is generated at the previous GenerateScatterRay call
        float3 throughput = normalTerminate ? path.thp : float3(0);       // the path throughput is updated in generate scatter ray.
        float3 bias = path.L;               // the L already contains the self emission at the hit vertex
        //writeScreenInferenceQuery(scatterOrigin, scatterDir, throughput, bias);
        writeScreenInferenceQuery(sd, throughput, bias);
    }

    const float3 queryRadiance = path.L;
    const uint inferencePathLength = path.length;
    logInferencePathLength(inferencePathLength);
    /* Path enlongation! (for RR or terminate heuristic) */
    if (!gIsTrainingPass) return queryRadiance;
    // save outputs at the query vertex

    // basically, if we needs to query the NRC, we use this flag and adds a query to the structured buffer.
    NRCRayFootprint suffixRayFootprint = { 0, 0, true, true };
    bool needsQuery = pathType == NRCPathType::TrainingPath;
    const uint kMaxTrainingSuffixIter = pathType == NRCPathType::TrainingPathRR ?
        NRC_MAX_TRAINING_RR_BOUNCES + 1: NRC_MAX_TRAINING_BOUNCES + 1;
    if (!lastScatterMissed) { 
        [loop]
        for(uint depth = 0; depth < kMaxTrainingSuffixIter; depth++)
        {
            {   // do a training sample snapshot here
                pathRecords[path.length] = generatePathVertexRecord(path, sd);
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
                || path.length - inferencePathLength >= kMaxTrainingSuffixIter - 1
                || path.nonSpecularBounces > kMaxNonSpecularBounces;

            // this "if" means we only takes the contribution from light sources into account.  [kUseMIS is enabled by default so ignore this]
            if (isLastVertex && (kUseNEE && !kUseMIS && path.isLightSamplable())) { needsQuery = false; break; }
            // terminate path at the first non specular vertex, after the path became specular. [kDisableCaustics is disabled by default]
            if (kDisableCaustics && path.isSpecular() && path.nonSpecularBounces > 0) { needsQuery = false; break; }
            // this "if" miss the next ray according to bsdf importance sampling is missed.
            if (!traceScatterRay(path.origin, path.dir, path.interiorList, path.hit)) { needsQuery = false; break; }

            const bool validHit = path.hit.isValid();

            if (validHit)
            {   // Handle current intersection, update path data & shading data for next scatter. Here the emissive radiance is added to path contribution.
                // path length is increased here
                float3 prevPos = sd.posW;
                handleHit(pt, sd, path);
                suffixRayFootprint = suffixRayFootprint.nextBounce(prevPos, sd.posW, path.pdf, sd.NdotV);
                debugRayFootprint(path, suffixRayFootprint);
}
            else
            {   // add MIS weighted contribution from environment map.
                lastScatterMissed = true;
                handleMiss(pt, path);
                needsQuery = false;
            }

            if (!validHit) break;
            if (isLastVertex) break;
            if (needsQuery && suffixRayFootprint.terminate()) break;
        }
    }

    logSuffixPathLength(path.length - inferencePathLength);
    int queryIndex = -1;
    if (needsQuery) {
        queryIndex = (int)addTrainingQuery(sd);
    }
    print("current query index: ", queryIndex);

    for(uint i = 0; i < path.length + (int)lastScatterMissed; i++) {
        // from 0 to length-1 is indeed the best choice, for either [end with RR] or [missed] or [max bounces limit reached]
        // however, when [missed] the length-th record could actually be used.
        float3 sampleL = (path.L - pathRecords[i].L) / pathRecords[i].thp;
        float3 sampleFactor = path.thp / pathRecords[i].thp;
        addTrainingSample(sampleFactor, sampleL, pathRecords[i].query, queryIndex);   
    }
    return queryRadiance;
}

float3 visualizeNRC(ShadingData sd){
    //float3 primaryRayDir = -sd.fromLocal(sd.V);
    writeScreenInferenceQuery(sd, float3(1.0), float3(0.0));
    return float3(0);
}
