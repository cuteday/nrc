#pragma once
#include "Falcor.h"
#include "FalcorExperimental.h"
#include "Utils/Sampling/SampleGenerator.h"

using namespace Falcor;

class RTAO : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<RTAO>;

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual std::string getDesc() override;
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    RTAO() = default;
    RTAO(const Dictionary& dict);

    // rtao parameters
    float mAoRadius = 0.25f;
    float mMinT = 0.0001f;
    uint mSampleCount = 8;

    uint mFrameCount = 0;

    Scene::SharedPtr mpScene = nullptr;

    SampleGenerator::SharedPtr mpSampleGenerator = nullptr;

    RtProgram::SharedPtr mpProgram = nullptr;
    RtProgramVars::SharedPtr mpProgramVars = nullptr;
};
