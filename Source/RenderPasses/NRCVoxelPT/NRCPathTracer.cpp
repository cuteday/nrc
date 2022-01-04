#include "NRCPathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Scene/HitInfo.h"
#include <sstream>
#include "Parameters.h"

namespace
{
    using namespace NRC;

    const char kShaderFile[] = "RenderPasses/NRCVoxelPT/PathTracer.rt.slang";
    const char kCompositeShaderFile[] = "RenderPasses/NRCVoxelPT/Composite.cs.slang";
    const char kParameterBlockName[] = "gData";

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    // The payload for the scatter rays is 8-12B.
    // The payload for the shadow rays is 4B.
    const uint32_t kMaxPayloadSizeBytes = HitInfo::kMaxPackedSizeInBytes;
    const uint32_t kMaxAttributeSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;

    // Render pass output channels.
    const std::string kColorOutput = "color";
    const std::string kAlbedoOutput = "albedo";
    const std::string kTimeOutput = "time";
    const std::string kNRCResultOutput = "result";
    const std::string kNRCFactorOutput = "factor";

    const Gui::DropdownList kNRCVisualizeModeList = {
        {(uint32_t)NRCVisualizeMode::Result, "composited radiance"},
        {(uint32_t)NRCVisualizeMode::Radiance, "queried radiance contribution"},
        {(uint32_t)NRCVisualizeMode::Factor, "factor of radiance contribution"},
        {(uint32_t)NRCVisualizeMode::Bias, "bias of radiance"},
        {(uint32_t)NRCVisualizeMode::Reflectance, "reflectance diffuse + specular"}
    };

    const Falcor::ChannelList kOutputChannels =
    {
        { kColorOutput,     "gOutputColor",               "Output color (linear) with contribution from NRC excluded", true /* optional */                              },
        { kAlbedoOutput,    "gOutputAlbedo",              "Surface albedo (base color) or background color", true /* optional */    },
        { kTimeOutput,      "gOutputTime",                "Per-pixel execution time", true /* optional */, ResourceFormat::R32Uint  },
        { kNRCResultOutput, "gOutputResult",              "NRC predicted radiance composited with outputColor", true                }
    };
};

using namespace NRC;
const char* NRCPathTracer::sDesc = "NRC path tracer";

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("NRCVoxelPT", NRCPathTracer::sDesc, NRCPathTracer::create);
}

NRCPathTracer::SharedPtr NRCPathTracer::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new NRCPathTracer(dict));
}

NRCPathTracer::NRCPathTracer(const Dictionary& dict)
    : PathTracer(dict, kOutputChannels)
{
    mTracer.pNRCPixelStats = NRCPixelStats::create();
    mHaltonSampler = HaltonSamplePattern::create(0);
    assert(mTracer.pNRCPixelStats);
}

/* under testing process */
bool NRCPathTracer::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint2 targetDim = renderData.getDefaultTextureDims();
    if (targetDim.x * targetDim.y > Parameters::max_inference_query_size) {
        logFatal("Screen size exceeds maximum inference restriction");
    }
    bool state = PathTracer::beginFrame(pRenderContext, renderData);
    if (!state) return false;
    if (!mNRC.pNRC) {
        mNRC.pNRC = NRC::NRCVoxelInterface::SharedPtr(new NRC::NRCVoxelInterface());
        mNRC.pNetwork = mNRC.pNRC->mNetwork;
        uint3 voxel_size = Parameters::voxel_param.voxel_size;
        uint num_voxels = voxel_size.x * voxel_size.y * voxel_size.z;
        logInfo("NRCVoxelPT::Initialize rendering resources for " + std::to_string(num_voxels) + " voxels!");
        mNRCVoxel.nVoxels = num_voxels;
    }

    if (!mNRC.pTrainingRadianceQuery) {
        /* there are 3 ways to create a structured buffer shader resource */
        mNRC.pTrainingRadianceQuery = Buffer::createStructured(sizeof(NRC::RadianceQuery), Parameters::max_training_query_size,
            Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
        if (mNRC.pTrainingRadianceQuery->getStructSize() != sizeof(NRC::RadianceQuery)) // check struct size to avoid alignment problems (?)
            throw std::runtime_error("Structure buffer size mismatch: training query");
        mNRC.pTrainingRadianceSample = Buffer::createStructured(sizeof(NRC::RadianceSample), Parameters::max_training_sample_size,
            Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
        if (mNRC.pTrainingRadianceSample->getStructSize() != sizeof(NRC::RadianceSample))
            throw std::runtime_error("Structure buffer size mismatch: training record");
        mNRC.pInferenceRadianceQuery = Buffer::createStructured(sizeof(NRC::RadianceQuery), Parameters::max_inference_query_size,
            Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
        if (mNRC.pInferenceRadianceQuery->getStructSize() != sizeof(NRC::RadianceQuery))
            throw std::runtime_error("Structure buffer size mismatch: inference query");
        mNRC.pSharedCounterBuffer = Buffer::createStructured(sizeof(uint32_t), 4,
            Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess);
        //mNRC.pInferenceRadiancePixel = Buffer::createStructured(sizeof(uint2), Parameters::max_inference_query_size,
        //    Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
        //if (mNRC.pInferenceRadiancePixel->getStructSize() != sizeof(uint2))
        //    throw std::runtime_error("Structure buffer size mismatch: inference pixel");
    }

    if (!mNRCVoxel.pInferenceQueryCounter) {
        // initialize counters!
        mNRCVoxel.pInferenceQueryCounter = Buffer::createStructured(sizeof(uint32_t), mNRCVoxel.nVoxels,
            Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
        mNRCVoxel.pTrainingQueryCounter = Buffer::createStructured(sizeof(uint32_t), mNRCVoxel.nVoxels,
            Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
        mNRCVoxel.pTrainingSampleCounter = Buffer::createStructured(sizeof(uint32_t), mNRCVoxel.nVoxels,
            Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
    }

    if (!mScreen.pScreenQueryBias || mScreen.pScreenQueryBias->getWidth() != targetDim.x || mScreen.pScreenQueryBias->getHeight() != targetDim.y) {
        mScreen.pScreenQueryBias = Texture::create2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, 1, 1,
            nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mScreen.pScreenQueryFactor = Texture::create2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, 1, 1,
            nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mScreen.pScreenResult = Texture::create2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, 1, 1,
            nullptr, ResourceBindFlags::Shared | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mScreen.pScreenQueryReflectance = Texture::create2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, 1, 1,
            nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

        // also register these resource to NRCInterface again
        mNRC.pNRC->registerNRCResources(mNRC.pInferenceRadianceQuery, mScreen.pScreenResult, mNRC.pTrainingRadianceQuery, mNRC.pTrainingRadianceSample,
            mNRCVoxel.pInferenceQueryCounter, mNRCVoxel.pTrainingSampleCounter, mNRCVoxel.pTrainingQueryCounter);
    }
    if (mNRCOptionChanged) {
        if (mNRC.enableNRC)
            mTracer.pProgram->addDefine("NRC_ENABLE");
        else mTracer.pProgram->removeDefine("NRC_ENABLE");
        mNRCOptionChanged = false;
    }
    pRenderContext->clearUAVCounter(mNRC.pTrainingRadianceQuery, 0);
    pRenderContext->clearUAVCounter(mNRC.pTrainingRadianceSample, 0);
    pRenderContext->clearUAVCounter(mNRC.pInferenceRadianceQuery, 0);

    pRenderContext->clearUAV(mNRCVoxel.pInferenceQueryCounter->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mNRCVoxel.pTrainingQueryCounter->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mNRCVoxel.pTrainingSampleCounter->getUAV().get(), uint4(0));
    {
        //pRenderContext->flush(true);
        //cudaMemset(mNRCVoxel.pInferenceQueryCounter->getCUDADeviceAddress(), 0, sizeof(uint32_t) * mNRCVoxel.nVoxels);
        //cudaMemset(mNRCVoxel.pTrainingQueryCounter->getCUDADeviceAddress(), 0, sizeof(uint32_t) * mNRCVoxel.nVoxels);
        //cudaMemset(mNRCVoxel.pTrainingSampleCounter->getCUDADeviceAddress(), 0, sizeof(uint32_t) * mNRCVoxel.nVoxels);
        //cudaDeviceSynchronize();
    }
    //
    mTracer.pNRCPixelStats->beginFrame(pRenderContext, renderData.getDefaultTextureDims());
    return state;
}

void NRCPathTracer::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    mTracer.pNRCPixelStats->endFrame(pRenderContext);
    PathTracer::endFrame(pRenderContext, renderData);
}

void NRCPathTracer::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    PathTracer::setScene(pRenderContext, pScene);
    if (mpScene)
    {
        if (mpScene->hasGeometryType(Scene::GeometryType::Procedural))
        {
            logWarning("This render pass only supports triangles. Other types of geometry will be ignored.");
        }

        // Create ray tracing program.
        RtProgram::Desc desc;
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(kMaxAttributeSizeBytes);
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
        desc.addDefines(mpScene->getSceneDefines());
        desc.addDefine("MAX_BOUNCES", std::to_string(mSharedParams.maxBounces));
        desc.addDefine("SAMPLES_PER_PIXEL", std::to_string(mSharedParams.samplesPerPixel));

        // NRC related parameters definations
        assert(mNRC.max_training_rr_bounces >= mNRC.max_training_bounces);
        desc.addDefine("NRC_MAX_TRAINING_BOUNCES", std::to_string(mNRC.max_training_bounces));
        desc.addDefine("NRC_MAX_TRAINING_RR_BOUNCES", std::to_string(mNRC.max_training_rr_bounces));
        desc.addDefine("NRC_MAX_INFERENCE_BOUNCES", std::to_string(mNRC.max_inference_bounces));

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(kRayTypeScatter, desc.addMiss("scatterMiss"));
        sbt->setMiss(kRayTypeShadow, desc.addMiss("shadowMiss"));
        sbt->setHitGroupByType(kRayTypeScatter, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("scatterClosestHit", "scatterAnyHit"));
        sbt->setHitGroupByType(kRayTypeShadow, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("", "shadowAnyHit"));

        mTracer.pProgram = RtProgram::create(desc);

        mCompositePass = ComputePass::create(kCompositeShaderFile, "main");
    }
}

void NRCPathTracer::renderUI(Gui::Widgets& widget)
{
    PathTracer::renderUI(widget);
    if (widget.checkbox("Enable NRC", mNRC.enableNRC)) {
        mNRCOptionChanged = true;
    }
    if(mNRC.enableNRC){
        widget.checkbox("Enable training", mNRC.pNRC->enableTraining());
        widget.checkbox("Enable inference", mNRC.pNRC->enableInference());

        if (auto group = widget.group("NRC Lowlevel Params")) {
            if (widget.var("Max inference bounces", mNRC.max_inference_bounces, 3, 15, 1)
                || widget.var("Max training suffix bounces", mNRC.max_training_bounces, 3, 15, 1)
                || widget.var("Max RR suffix bounces", mNRC.max_training_rr_bounces, 3, 15, 1)) {
                mOptionsChanged = true;
            }
        }
        widget.var("Terminate threshold inference", mNRC.footprint_thres_inference, 0.f, 15.f, 0.001f);
        widget.var("Terminate threshold suffix", mNRC.foorprint_thres_suffix, 0.f, 50.f, 0.001f);
        widget.var("Blend factor", mNRC.prob_blend_nrc, 0.f, 1.f, 0.001f);

        if (auto group = widget.group("NRC Debug")) {
            // widget.group creates a sub widget.
            mTracer.pNRCPixelStats->renderUI(group);
            widget.checkbox("visualize NRC", mNRC.visualizeNRC);
            widget.tooltip("Query the NRC at primary vertices.");
            widget.tooltip("visualize factor of the NRC contribution query");
            widget.dropdown("visualize mode", kNRCVisualizeModeList, mNRC.visualizeMode);
 
        }
        if (auto group = widget.group("Network Params")) {
            if (widget.button("reset network")) {
                mNRC.pNRC->resetParameters();
            }
            widget.var("Learning rate", mNRC.pNetwork->learningRate(), 0.f, 1e-2f, 1e-5f);
        }
    }

}

void NRCPathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Call shared pre-render code.
    if (!beginFrame(pRenderContext, renderData)) return;

    // Set compile-time constants.
    RtProgram::SharedPtr pProgram = mTracer.pProgram;
    setStaticParams(pProgram.get());

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    pProgram->addDefines(getValidResourceDefines(mInputChannels, renderData));
    pProgram->addDefines(getValidResourceDefines(mOutputChannels, renderData));

    if (mUseEmissiveSampler)
    {
        // Specialize program for the current emissive light sampler options.
        assert(mpEmissiveSampler);
        if (pProgram->addDefines(mpEmissiveSampler->getDefines())) mTracer.pVars = nullptr;
    }

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars) prepareVars();
    assert(mTracer.pVars);

    // Set shared data into parameter block.
    setTracerData(renderData);

    // Set NRC data and parameters
    setNRCData(renderData);

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            auto var = mTracer.pVars->getRootVar();
            var[desc.texname] = renderData[desc.name]->asTexture();
        }
    };
    for (auto channel : mInputChannels) bind(channel);
    for (auto channel : mOutputChannels) bind(channel);

    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    assert(targetDim.x > 0 && targetDim.y > 0);

    auto vars = mTracer.pVars->getRootVar();
    mpPixelDebug->prepareProgram(pProgram, vars);
    mpPixelStats->prepareProgram(pProgram, vars);
    mTracer.pNRCPixelStats->prepareProgram(pProgram, vars);

    // Spawn the rays for inference.
    {
        PROFILE("NRCPathTracer::execute()_RayTrace_Inference");
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));
    }
    // "Enlong" the training suffix
    if (mNRC.enableNRC) {
        {
            PROFILE("NRCPathTracer::execute()_RayTrace_TrainingSuffix");
            mTracer.pVars["NRCDataCB"]["gIsTrainingPass"] = true;
            mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim / Parameters::trainingPathStride, 1));
        }
        // well now it seems the raytracing shader invocation is asynchronous, do synchronization step here.
        //gpDevice->getRenderContext()->flush(true);
        pRenderContext->flush(true);
        {
            PROFILE("NRCPathTracer::execute()_CUDA_Prepare_Resources");
            mNRC.pNRC->prepare();
        }
        {
            // this takes ~10ms
            PROFILE("NRCPathTracer::execute()_CUDA_Network_Inference");
            mNRC.pNRC->inferenceFrame();
        }
        {
            // this takes ~3ms
            PROFILE("NRCPathTracer::execute()_CUDA_Network_Training");
            // no, we make training process an ansynchronous step.
            mNRC.pNRC->trainFrame();
        }
        // here we wait until all emitted cuda commands finish. 
        {
            PROFILE("NRCPathTracer::execute()_Composite_Outputs");
            mCompositePass->execute(pRenderContext, uint3(targetDim, 1));
        }
        cudaDeviceSynchronize();
    }
    // Call shared post-render code.
    endFrame(pRenderContext, renderData);
}

RenderPassReflection NRCPathTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector = PathTracer::reflect(compileData);
    return reflector;
}

void NRCPathTracer::prepareVars()
{
    assert(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    bool success = mpSampleGenerator->setShaderData(var);
    if (!success) throw std::exception("Failed to bind sample generator");

    // Create parameter block for shared data.
    ProgramReflection::SharedConstPtr pReflection = mTracer.pProgram->getReflector();
    ParameterBlockReflection::SharedConstPtr pBlockReflection = pReflection->getParameterBlock(kParameterBlockName);
    assert(pBlockReflection);
    mTracer.pParameterBlock = ParameterBlock::create(pBlockReflection);
    assert(mTracer.pParameterBlock);

    // Bind static resources to the parameter block here. No need to rebind them every frame if they don't change.
    // Bind the light probe if one is loaded.
    if (mpEnvMapSampler) mpEnvMapSampler->setShaderData(mTracer.pParameterBlock["envMapSampler"]);

    // Bind the parameter block to the global program variables.
    mTracer.pVars->setParameterBlock(kParameterBlockName, mTracer.pParameterBlock);

    // set some static parameters.

}

void NRCPathTracer::setTracerData(const RenderData& renderData)
{
    auto pBlock = mTracer.pParameterBlock;
    assert(pBlock);

    // Upload parameters struct.
    pBlock["params"].setBlob(mSharedParams);

    // Bind emissive light sampler.
    if (mUseEmissiveSampler)
    {
        assert(mpEmissiveSampler);
        bool success = mpEmissiveSampler->setShaderData(pBlock["emissiveSampler"]);
        if (!success) throw std::exception("Failed to bind emissive light sampler");
    }
}

void NRCPathTracer::setNRCData(const RenderData& renderData)
{
    // NRC related testing process
    auto pVars = mTracer.pVars;
    // width * height
    pVars["NRCDataCB"]["gNRCEnable"] = mNRC.enableNRC;
    pVars["NRCDataCB"]["gVisualizeMode"] = mNRC.visualizeNRC;
    pVars["NRCDataCB"]["gIsTrainingPass"] = false;      // reset this flag for next frame
    pVars["NRCDataCB"]["gNRCScreenSize"] = renderData.getDefaultTextureDims();
    pVars["NRCDataCB"]["gNRCTrainingPathOffset"] = uint2((1.f+mHaltonSampler->next()) * (float2)Parameters::trainingPathStride);
    pVars["NRCDataCB"]["gNRCTrainingPathStride"] = Parameters::trainingPathStride;
    pVars["NRCDataCB"]["gNRCTrainingPathStrideRR"] = Parameters::trainingPathStrideRR;
    pVars["NRCDataCB"]["gNRCAbsorptionProb"] = mNRC.prob_rr_suffix_absorption;
    pVars["NRCDataCB"]["gNRCBlendProb"] = mNRC.prob_blend_nrc;

    pVars["NRCDataCB"]["gFootprintThresInference"] = mNRC.footprint_thres_inference;
    pVars["NRCDataCB"]["gFootprintThresSuffix"] = mNRC.foorprint_thres_suffix;
    // scene AABB for normalizing coordinates
    pVars["NRCDataCB"]["gSceneAABBCenter"] = mpScene->getSceneBounds().center();
    pVars["NRCDataCB"]["gSceneAABBExtent"] = mpScene->getSceneBounds().extent();
    // voxel related parameters
    pVars["VoxelDataCB"]["gVoxelSize"] = Parameters::voxel_param.voxel_size;
    // set textures & buffers (defined in NrC.slang)
    pVars["gScreenQueryFactor"] = mScreen.pScreenQueryFactor;
    pVars["gScreenQueryBias"] = mScreen.pScreenQueryBias;
    pVars["gScreenQueryReflectance"] = mScreen.pScreenQueryReflectance;

    pVars["gInferenceRadianceQuery"] = mNRC.pInferenceRadianceQuery;
    pVars["gTrainingRadianceQuery"] = mNRC.pTrainingRadianceQuery;
    pVars["gTrainingRadianceSample"] = mNRC.pTrainingRadianceSample;

    pVars["gVoxelInferenceQueryCounter"] = mNRCVoxel.pInferenceQueryCounter;
    pVars["gVoxelTrainingSampleCounter"] = mNRCVoxel.pTrainingSampleCounter;
    pVars["gVoxelTrainingQueryCounter"] = mNRCVoxel.pTrainingQueryCounter;

    mCompositePass["CompositeCB"]["gVisualizeMode"] = mNRC.visualizeMode;
    mCompositePass["CompositeCB"]["gReflectanceFact"] = (bool)REFLECTANCE_FACT;
    mCompositePass["factor"] = mScreen.pScreenQueryFactor;
    mCompositePass["bias"] = mScreen.pScreenQueryBias;
    mCompositePass["radiance"] = mScreen.pScreenResult;
    mCompositePass["reflectance"] = mScreen.pScreenQueryReflectance;
    mCompositePass["output"] = renderData[kNRCResultOutput]->asTexture();
}
