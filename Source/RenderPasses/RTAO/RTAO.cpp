#include "RTAO.h"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{
    const char kDesc[] = "Ambient occlusion with ray tracing";

    // constant strings for uniforms and texture names here
    //const std::string kColorIn = "colorIn";
    //const std::string kColorOut = "colorOut";
    //const std::string kNormals = "normals";
    //const std::string kAoMap = "aoMap";
    //const std::string kWorldPos = "positions";

    const std::string kProgramRaytraceFile = "RenderPasses/RTAO/RTAO.rt.slang";

    // parameters
    const std::string kMinT = "minT";
    const std::string kAoRadius = "aoRadius";
    const std::string kSampleCount = "sampleCount";

    // we use renderpass helper functions since raytrace shader needs UAV textures, which supports r32f format only
    // these helper functions sets UAV flags and r32f formats for you (*/ω＼*)
    const ChannelList kInputChannels = {
        // see ChannelList defination...
        {"colorIn", "gColorIn", "Input color buffer"},
        {"normalW", "gNormals", "World space normals in[0, 1]"},
        {"posW", "gWorldPos", "World positions [xyz] and foreground flag [w]"}
    };

    const ChannelList kOutputChannels = {
        {"colorOut", "gColorOut", "Output target buffer"},
        {"aoMap", "gAoMap", "Ambient occlusion factor"}
    };

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 80u;
    const uint32_t kMaxAttributeSizeBytes = 8u;
    const uint32_t kMaxRecursionDepth = 2u;
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("RTAO", kDesc, RTAO::create);
}

RTAO::RTAO(const Dictionary& dict)
{
    // parse passed dict
    for (const auto& [key, value] : dict) {
        if (key == kAoRadius) mAoRadius = value;
        if (key == kSampleCount) mSampleCount = value;
        if (key == kMinT) mMinT = value;
    }
    // create a sample generator (and bind it to shader later)
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);
    assert(mpSampleGenerator);
}

RTAO::SharedPtr RTAO::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new RTAO(dict));
    return pPass;
}

std::string RTAO::getDesc() { return kDesc; }

Dictionary RTAO::getScriptingDictionary()
{
    Dictionary d;
    d[kAoRadius] = mAoRadius;
    d[kSampleCount] = mSampleCount;
    d[kMinT] = mMinT;
    return d;
}

RenderPassReflection RTAO::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    //reflector.addOutput(kColorOut, "Color buffer with AO").bindFlags(ResourceBindFlags::UnorderedAccess);
    //reflector.addOutput(kAoMap, "AO factor").bindFlags(ResourceBindFlags::UnorderedAccess);

    //reflector.addInput(kColorIn, "Color buffer");
    //reflector.addInput(kNormals, "World space normals in [0, 1]");
    //reflector.addInput(kWorldPos, "World positions [xyz] and foreground flag [w]");

    return reflector;
}

void RTAO::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) {
        // scene not set (neither the shader program)
        // if we can, clear all the output textures [pRenderContext->clearTexture()]
        return;
    }

    // set constant buffer variables
    auto var = mpProgramVars->getRootVar();
    var["RayTraceCB"]["gFrameCount"] = mFrameCount;
    var["RayTraceCB"]["gAoRadius"] = mAoRadius;
    var["RayTraceCB"]["gMinT"] = mMinT;
    var["RayTraceCB"]["gSampleCount"] = mSampleCount;

    //var["gAoMap"] = renderData[kAoMap]->asTexture();
    //var["gColorOut"] = renderData[kColorOut]->asTexture();
    //var["gNormals"] = renderData[kNormals]->asTexture();
    //var["gColorIn"] = renderData[kColorIn]->asTexture();
    //var["gWorldPos"] = renderData[kWorldPos]->asTexture();

    auto bind = [&](const ChannelDesc& desc) {
        var[desc.texname] = renderData[desc.name]->asTexture();
    };
    for (auto& channel : kInputChannels)bind(channel);
    for (auto& channel : kOutputChannels)bind(channel);

    // get screen dimensions for ray dispatch
    const uint2 targetDim = renderData.getDefaultTextureDims();
    assert(targetDim.x > 0 && targetDim.y > 0);

    // this call to raytrace builds the accel structures for the scene, then spawn the rays
    mpScene->raytrace(pRenderContext, mpProgram.get(), mpProgramVars, uint3(targetDim, 1));

    mFrameCount++;
}

void RTAO::renderUI(Gui::Widgets& widget)
{
    // tune parameters here
    widget.text("Hello DXR!");
    widget.var("Sample radius", mAoRadius, 0.001f, 10.0f, 0.001f);
    widget.var("Sample count", mSampleCount, 0u, 32u, 1u);
    widget.slider("Minimun ray distance", mMinT, 0.0001f, 0.002f);
}

void RTAO::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    // clear data for previous scene is a good convention > <
    mpProgram = nullptr;
    mpProgramVars = nullptr;

    mpScene = pScene;
    // RenderPass::setScene(pRenderContext, pScene); // inherited nothing here

    if (mpScene) {
        // usually create programe on resetting scene!
        RtProgram::Desc desc;
        // the following attributes must be set!
        desc.addShaderLibrary(kProgramRaytraceFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(kMaxAttributeSizeBytes);
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
        desc.addDefines(mpScene->getSceneDefines());
        desc.addDefines(mpSampleGenerator->getDefines());

        RtBindingTable::SharedPtr sbt = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("miss"));
        sbt->setHitGroupByType(0, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("closestHit", "anyHit"));

        mpProgram = RtProgram::create(desc);
        mpProgramVars = RtProgramVars::create(mpProgram, sbt);

        // bind sample generator to shared data
        auto var = mpProgramVars->getRootVar();
        assert(mpSampleGenerator->setShaderData(var));
    }
}
