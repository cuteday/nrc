#include "NRCPathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Scene/HitInfo.h"
#include <sstream>

namespace
{
    const char kShaderFile[] = "RenderPasses/NRCPathTracer/PathTracer.rt.slang";
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

    const Falcor::ChannelList kOutputChannels =
    {
        { kColorOutput,     "gOutputColor",               "Output color (linear)", true /* optional */                              },
        { kAlbedoOutput,    "gOutputAlbedo",              "Surface albedo (base color) or background color", true /* optional */    },
        { kTimeOutput,      "gOutputTime",                "Per-pixel execution time", true /* optional */, ResourceFormat::R32Uint  },
    };
};

const char* NRCPathTracer::sDesc = "NRC path tracer";

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("NRCPathTracer", NRCPathTracer::sDesc, NRCPathTracer::create);
}

NRCPathTracer::SharedPtr NRCPathTracer::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new NRCPathTracer(dict));
}

NRCPathTracer::NRCPathTracer(const Dictionary& dict)
    : PathTracer(dict, kOutputChannels)
{
}

/* under testing process */
bool NRCPathTracer::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint2 targetDim = renderData.getDefaultTextureDims();
    if (targetDim.x * targetDim.y > mNRC.maximum_inference_buffer_size) {
        logFatal("Screen size exceeds maximum inference restriction");
    }
    bool state = PathTracer::beginFrame(pRenderContext, renderData);
    if (!state) return false;
    if (!mNRC.pNRC) {
        mNRC.pNRC = NRC::NRCInterface::SharedPtr(new NRC::NRCInterface());
    }
    if (!mNRC.pTrainingRadianceQuery) {
        /* there are 3 ways to create a structured buffer shader resource */
        //mNRC.pSample = Buffer::createStructured(mTracer.pProgram.get(), "gSample", 2000);
        //mNRC.pSample = Buffer::createStructured(mTracer.pVars.getRootVar()["gSample"], 2000);
        mNRC.pTrainingRadianceQuery = Buffer::createStructured(sizeof(NRC::RadianceQuery), mNRC.maximum_training_buffer_size,
            Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
        if (mNRC.pTrainingRadianceQuery->getStructSize() != sizeof(NRC::RadianceQuery))
            throw std::runtime_error("Structure buffer size mismatch: training query");
    }
    if (!mNRC.pTrainingRadianceRecord) {
        mNRC.pTrainingRadianceRecord = Buffer::createStructured(sizeof(NRC::RadianceRecord), mNRC.maximum_training_buffer_size,
            Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
        if (mNRC.pTrainingRadianceRecord->getStructSize() != sizeof(NRC::RadianceRecord))
            throw std::runtime_error("Structure buffer size mismatch: training record");
    }
    if (!mNRC.pInferenceRadiaceQuery) {
        mNRC.pInferenceRadiaceQuery = Buffer::createStructured(sizeof(NRC::RadianceQuery), mNRC.maximum_inference_buffer_size,
            Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
        if (mNRC.pInferenceRadiaceQuery->getStructSize() != sizeof(NRC::RadianceQuery))
            throw std::runtime_error("Structure buffer size mismatch: inference query");
    }
    //pRenderContext->clearUAVCounter(mNRC.pTrainingRadianceQuery, 0);
    return state;
}

void NRCPathTracer::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
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

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(kRayTypeScatter, desc.addMiss("scatterMiss"));
        sbt->setMiss(kRayTypeShadow, desc.addMiss("shadowMiss"));
        sbt->setHitGroupByType(kRayTypeScatter, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("scatterClosestHit", "scatterAnyHit"));
        sbt->setHitGroupByType(kRayTypeShadow, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("", "shadowAnyHit"));

        mTracer.pProgram = RtProgram::create(desc);
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

    mpPixelDebug->prepareProgram(pProgram, mTracer.pVars->getRootVar());
    mpPixelStats->prepareProgram(pProgram, mTracer.pVars->getRootVar());

    // Spawn the rays.
    {
        PROFILE("NRCPathTracer::execute()_RayTrace");
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));
    }

    // Call shared post-render code.
    endFrame(pRenderContext, renderData);
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
    pVars["NRCDataCB"]["screenSize"] = renderData.getDefaultTextureDims();
    pVars["NRCDataCB"]["trainingPathOffset"] = uint2(std::rand() / (float)RAND_MAX * mNRC.trainingPathOffset.x,
        std::rand() / (float)RAND_MAX * mNRC.trainingPathOffset.y);
}

