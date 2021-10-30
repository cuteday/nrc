#include "RTLightingPass.h"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{
    const char kDesc[] = "Lighting pass using RTX shadowray tracing";

    const std::string kProgramRaytraceFile = "RenderPasses/RTLightingPass/RTLightingPass.rt.slang";

    const ChannelList kInputChannels = {
        {"shadingNormalW", "gShadingNormals", ""},
        {"faceNormalW", "gFaceNormals", ""},
        {"posW", "gWorldPos", ""},
        {"diffuseOpacity", "gDiffuseOpacity", ""},
        {"specRough", "gSpecRough", ""},
        {"emissive", "gEmissive", ""},
        {"mtlExtraParams", "gMtlExtraParams", ""}
    };

    const ChannelList kOutputChannels = {
        {"colorOut", "gColorOut"}
    };

    const std::string kMinT = "minT";
    const std::string kDirectSampleCount = "directSampleCount";

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
    lib.registerClass("RTLightingPass", kDesc, RTLightingPass::create);
}

RTLightingPass::RTLightingPass(const Dictionary& dict)
{
    for (const auto& [key, value] : dict) {
        if (key == kMinT) mMinT = value;
    }
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);
    assert(mpSampleGenerator);
}

RTLightingPass::SharedPtr RTLightingPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new RTLightingPass(dict));
    return pPass;
}

std::string RTLightingPass::getDesc() { return kDesc; }

Dictionary RTLightingPass::getScriptingDictionary()
{
    Dictionary d;
    d[kMinT] = mMinT;
    d[kDirectSampleCount] = mDirectSampleCount;
    return d;
}

RenderPassReflection RTLightingPass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void RTLightingPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
        return;

    auto var = mpProgramVars->getRootVar();
    var["RayTraceCB"]["gFrameCount"] = mFrameCount;
    var["RayTraceCB"]["gMinT"] = mMinT;
    var["RayTraceCB"]["gDirectSampleCount"] = mDirectSampleCount;

    auto bind = [&](const ChannelDesc& desc) {
        var[desc.texname] = renderData[desc.name]->asTexture();
    };
    for (auto& channel : kInputChannels)bind(channel);
    for (auto& channel : kOutputChannels)bind(channel);

    const uint2 targetDim = renderData.getDefaultTextureDims();
    assert(targetDim.x > 0 && targetDim.y > 0);
    mpScene->raytrace(pRenderContext, mpProgram.get(), mpProgramVars, uint3(targetDim, 1));
    mFrameCount++;
}

void RTLightingPass::renderUI(Gui::Widgets& widget)
{
    widget.text("Hello DXR lighting!");
    widget.var("Minimum ray distance", mMinT, 0.0f, 0.02f, 0.0001f);
    widget.var("Direct lighting sample count", mDirectSampleCount, 0u, 64u, 1);
    widget.tooltip("Average weighting these direct samples, 0 means sampling each light once");
}

void RTLightingPass::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpProgram = nullptr;
    mpProgramVars = nullptr;

    mpScene = pScene;
    if (mpScene) {
        RtProgram::Desc desc;
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

        auto var = mpProgramVars->getRootVar();
        mpSampleGenerator->setShaderData(var);

        // conventional routine: setting up environment map lighting
        //const auto& pEnvMap = mpScene->getEnvMap();
        //if (pEnvMap && (!mpEnvMapLighting || mpEnvMapLighting->getEnvMap() != pEnvMap)) {
        //    mpEnvMapLighting = EnvMapLighting::create(pRenderContext, pEnvMap);
        //    mpEnvMapLighting->setShaderData(var["gEnvMap"]);
        //}
        //else if (!pEnvMap) {
        //    mpEnvMapLighting = nullptr;
        //}
    }
}

