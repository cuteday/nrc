#pragma once
#include "Falcor.h"
#include "RenderPasses/Shared/PathTracer/PathTracer.h"
#include "NRC.h"
#include "Debug/NRCPixelStats.h"

using namespace Falcor;


/** Forward path tracer using a NRC.

    The path tracer has a loop over the path vertices in the raygen shader.
    The kernel terminates when all paths have terminated.

    This pass implements a forward path tracer with next-event estimation,
    Russian roulette, and multiple importance sampling (MIS) with sampling
    of BRDFs and light sources.
*/
class NRCPathTracer : public PathTracer
{
public:
    using SharedPtr = std::shared_ptr<NRCPathTracer>;

    static SharedPtr create(RenderContext* pRenderContext, const Dictionary& dict);

    virtual std::string getDesc() override { return sDesc; }
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;

    static const char* sDesc;

private:
    NRCPathTracer(const Dictionary& dict);

    //~NRCPathTracer();

    // these 2 functions are called within execute()
    bool beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void endFrame(RenderContext* pRenderContext, const RenderData& renderData);

    void recreateVars() override { mTracer.pVars = nullptr; }
    void prepareVars();
    void setTracerData(const RenderData& renderData);
    void setNRCData(const RenderData& renderData);

    // Ray tracing program.
    struct
    {
        RtProgram::SharedPtr pProgram;
        NRCPixelStats::SharedPtr pNRCPixelStats;
        RtBindingTable::SharedPtr pBindingTable;
        RtProgramVars::SharedPtr pVars;
        ParameterBlock::SharedPtr pParameterBlock;      ///< ParameterBlock for all data.
    } mTracer;

    // Neural radiance cache parameters and data fields
    struct {
        NRC::NRCInterface::SharedPtr pNRC = nullptr;

        bool enableNRC = false;
        float prob_rr_suffix_absorption = 0.2f;
        float terminate_footprint_thres = 50.f;
        int max_training_bounces = 5;               // max path segments for training suffix
        int max_training_rr_bounces = 10;
        int max_inference_bounces = 5;

        Buffer::SharedPtr pTrainingRadianceQuery = nullptr;
        Buffer::SharedPtr pTrainingRadianceSample = nullptr;
        Buffer::SharedPtr pInferenceRadiaceQuery = nullptr;
        Buffer::SharedPtr pSharedCounterBuffer = nullptr;
        Texture::SharedPtr pScreenQueryFactor = nullptr;
        Texture::SharedPtr pScreenQueryBias = nullptr;
        Texture::SharedPtr pScreenResult = nullptr;

    } mNRC;

    ComputePass::SharedPtr mCompositePass;
};
