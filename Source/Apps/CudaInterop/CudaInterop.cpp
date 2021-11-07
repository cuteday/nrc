#include "CudaInterop.h"
#include "CopySurface.h"

void CudaInterop::onLoad(RenderContext* pRenderContext)
{
    // Create our input and output textures
    mpInputTex = Texture::createFromFile("smoke-puff.png", false, false, ResourceBindFlags::Shared);
    mWidth = mpInputTex->getWidth();
    mHeight = mpInputTex->getHeight();
    mpOutputTex = Texture::create2D(mWidth, mHeight, mpInputTex->getFormat(), 1, 1, nullptr, ResourceBindFlags::Shared | ResourceBindFlags::ShaderResource);

    // Define our usage flags and then map the textures to CUDA surfaces. Surface values of 0
    // indicate an error during mapping. We need to cache mInputSurf and mOutputSurf as
    // mapTextureToSurface() can only be called once per resource.
    uint32_t usageFlags = cudaArrayColorAttachment;
    mInputSurf = FalcorCUDA::mapTextureToSurface(mpInputTex, usageFlags);
    if (mInputSurf == 0)
    {
        logError("Input texture to surface mapping failed");
        return;
    }
    mOutputSurf = FalcorCUDA::mapTextureToSurface(mpOutputTex, usageFlags);
    if (mOutputSurf == 0)
    {
        logError("Output texture to surface mapping failed");
        return;
    }
}

void CudaInterop::onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const Falcor::float4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    // Call the CUDA kernel
    uint32_t format = (getFormatType(mpInputTex->getFormat()) == FormatType::Float) ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
    launchCopySurface(mInputSurf, mOutputSurf, mWidth, mHeight, format);
    pRenderContext->blit(mpOutputTex->getSRV(), pTargetFbo->getRenderTargetView(0));
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    // Initializes the CUDA driver API, which is required prior to any API calls.
    if (!FalcorCUDA::initCUDA())
    {
        logError("CUDA driver API initialization failed");
        exit(1);
    }

    CudaInterop::UniquePtr pRenderer = std::make_unique<CudaInterop>();
    SampleConfig config;
    config.windowDesc.title = "Falcor-Cuda Interop";
    config.windowDesc.resizableWindow = true;

    Sample::run(config, pRenderer);
}
