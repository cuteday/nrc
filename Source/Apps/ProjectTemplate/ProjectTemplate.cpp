#include "ProjectTemplate.h"
uint32_t mSampleGuiWidth = 250;
uint32_t mSampleGuiHeight = 200;
uint32_t mSampleGuiPositionX = 20;
uint32_t mSampleGuiPositionY = 40;

void ProjectTemplate::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Falcor", { 250, 200 });
    gpFramework->renderGlobalUI(pGui);
    w.text("Hello from ProjectTemplate");
    if (w.button("Click Here"))
    {
        msgBox("Now why would you do that?");
    }
}

void ProjectTemplate::onLoad(RenderContext* pRenderContext)
{
}

void ProjectTemplate::onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const float4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
}

void ProjectTemplate::onShutdown()
{
}

bool ProjectTemplate::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return false;
}

bool ProjectTemplate::onMouseEvent(const MouseEvent& mouseEvent)
{
    return false;
}

void ProjectTemplate::onHotReload(HotReloadFlags reloaded)
{
}

void ProjectTemplate::onResizeSwapChain(uint32_t width, uint32_t height)
{
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    ProjectTemplate::UniquePtr pRenderer = std::make_unique<ProjectTemplate>();
    SampleConfig config;
    config.windowDesc.title = "Falcor Project Template";
    config.windowDesc.resizableWindow = true;
    Sample::run(config, pRenderer);
    return 0;
}
