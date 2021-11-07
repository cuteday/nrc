#pragma once
#include "Falcor.h"
#include "RenderPasses/Shared/PathTracer/PathTracer.h"
#include "NRC.h"

using namespace Falcor;
using Falcor::uint;
using Falcor::uint2;
using Falcor::uint3;
using Falcor::uint4;

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
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;

    static const char* sDesc;

private:
    NRCPathTracer(const Dictionary& dict);

    // these 2 functions are called within execute()
    bool beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void endFrame(RenderContext* pRenderContext, const RenderData& renderData);

    void recreateVars() override { mTracer.pVars = nullptr; }
    void prepareVars();
    void setTracerData(const RenderData& renderData);

    // Ray tracing program.
    struct
    {
        RtProgram::SharedPtr pProgram;
        RtBindingTable::SharedPtr pBindingTable;
        RtProgramVars::SharedPtr pVars;
        ParameterBlock::SharedPtr pParameterBlock;      ///< ParameterBlock for all data.
    } mTracer;

    // Neural radiance cache
    struct {
        Buffer::SharedPtr pSample = nullptr;
        NRC::NRCInterface::SharedPtr pNRC = nullptr;

    } mNRC;
};
