#pragma once

#include "NRC.h"
#include "Falcor.h"
#include "RenderPasses/Shared/PathTracer/PathTracer.h"
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
        //NRC::NRCInterface::SharedPtr pNRC = nullptr;
        NRC::NRCVoxelInterface::SharedPtr pNRC = nullptr;
        //NRC::NRCNetwork::SharedPtr pNetwork = nullptr;
        NRC::VoxelNetwork::SharedPtr pNetwork = nullptr;

        bool enableNRC = true;
        bool visualizeNRC = false;
        uint32_t visualizeMode = 1;
        float prob_rr_suffix_absorption = 0.2f;
        float prob_blend_nrc = 0.2f;
        float footprint_thres_inference = 2.5f;
        float foorprint_thres_suffix = 20.f;

        int max_training_bounces = 5;               // max path segments for training suffix
        int max_training_rr_bounces = 10;
        int max_inference_bounces = 5;

        Buffer::SharedPtr pTrainingRadianceQuery = nullptr;
        Buffer::SharedPtr pTrainingRadianceSample = nullptr;
        Buffer::SharedPtr pInferenceRadianceQuery = nullptr;
        Buffer::SharedPtr pInferenceRadiancePixel = nullptr;
        Buffer::SharedPtr pSharedCounterBuffer = nullptr;
    } mNRC;

    struct {
        // voxel related
        uint nVoxels = 0;

        Buffer::SharedPtr* pTrainingQueryVoxel = nullptr;
        Buffer::SharedPtr* pTrainingSampleVoxel = nullptr;
        Buffer::SharedPtr* pInferenceQueryVoxel = nullptr;
        Buffer::SharedPtr* pInferencePixelVoxel = nullptr;

        Buffer::SharedPtr pInferenceQueryCounter = nullptr;
        Buffer::SharedPtr pTrainingSampleCounter = nullptr;
        Buffer::SharedPtr pTrainingQueryCounter = nullptr;
    } mNRCVoxel;

    struct {
        Texture::SharedPtr pScreenQueryFactor = nullptr;
        Texture::SharedPtr pScreenQueryBias = nullptr;
        Texture::SharedPtr pScreenQueryReflectance = nullptr;
        Texture::SharedPtr pScreenResult = nullptr;
    } mScreen;

    bool mNRCOptionChanged = true;
    ComputePass::SharedPtr mCompositePass;
    HaltonSamplePattern::SharedPtr mHaltonSampler;
};
