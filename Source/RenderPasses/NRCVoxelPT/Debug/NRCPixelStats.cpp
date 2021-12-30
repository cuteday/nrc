#include "stdafx.h"

#include "NRCPixelStats.h"
#include "../Parameters.h"

#include <sstream>
#include <iomanip>

namespace Falcor
{
    namespace
    {
        
    }

    NRCPixelStats::SharedPtr NRCPixelStats::create()
    {
        return SharedPtr(new NRCPixelStats());
    }

    NRCPixelStats::NRCPixelStats()
    {
    }

    void NRCPixelStats::beginFrame(RenderContext* pRenderContext, const uint2& frameDim)
    {
        // Prepare state.
        assert(!mRunning);
        mRunning = true;
        mWaitingForData = false;
        mFrameDim = frameDim;

        // Mark previously stored data as invalid. The config may have changed, so this is the safe bet.
        mStats = Stats();
        mStatsValid = false;
        mStatsBuffersValid = false;
        mRayCountTextureValid = false;

        if (mEnabled)
        {
            // Create parallel reduction helper.
            if (!mpParallelReduction)
            {
                mpParallelReduction = ComputeParallelReduction::create();
                mpReductionResult = Buffer::create((2 /* number of stats counters */) * sizeof(uint4), ResourceBindFlags::None, Buffer::CpuAccess::Read);
            }

            // Prepare stats buffers.
            if (!mpStatsInferencePathLength || mpStatsInferencePathLength->getWidth() != frameDim.x || mpStatsInferencePathLength->getHeight() != frameDim.y)
            {
                mpStatsInferencePathLength = Texture::create2D(frameDim.x, frameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                mpStatsSuffixPathLength = Texture::create2D(frameDim.x, frameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            }
            //pRenderContext->clearTexture(mpStatsInferencePathLength.get(), uint4(0));
            //pRenderContext->clearTexture(mpStatsSuffixPathLength.get(), uint4(0));
            pRenderContext->clearUAV(mpStatsSuffixPathLength->getUAV().get(), uint4(0));
            pRenderContext->clearUAV(mpStatsInferencePathLength->getUAV().get(), uint4(0));
        }
    }

    void NRCPixelStats::endFrame(RenderContext* pRenderContext)
    {
        assert(mRunning);
        mRunning = false;

        if (mEnabled)
        {
            // Create fence first time we need it.
            if (!mpFence) mpFence = GpuFence::create();

            // the parallel reduction uses a compute shader to execute computation.
            // note that *result pointer argument also needs to map from GPU memory to cpu (not a faster method...)
            mpParallelReduction->execute<uint4>(pRenderContext, mpStatsInferencePathLength, ComputeParallelReduction::Type::Sum, nullptr, mpReductionResult, 0 * sizeof(uint4));
            mpParallelReduction->execute<uint4>(pRenderContext, mpStatsSuffixPathLength, ComputeParallelReduction::Type::Sum, nullptr, mpReductionResult, 1 * sizeof(uint4));

            // Submit command list and insert signal.
            pRenderContext->flush(false);
            mpFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

            mStatsBuffersValid = true;
            mWaitingForData = true;
        }
    }

    void NRCPixelStats::prepareProgram(const Program::SharedPtr& pProgram, const ShaderVar& var)
    {
        assert(mRunning);

        if (mEnabled)
        {
            pProgram->addDefine("_NRC_PIXEL_STATS_ENABLED");
            var["gStatsInferencePathLength"] = mpStatsInferencePathLength;
            var["gStatsSuffixPathLength"] = mpStatsSuffixPathLength;
        }
        else
        {
            pProgram->removeDefine("_NRC_PIXEL_STATS_ENABLED");
        }
    }

    void NRCPixelStats::renderUI(Gui::Widgets& widget)
    {
        // Configuration.
        widget.checkbox("Ray stats", mEnabled);
        widget.tooltip("Collects ray tracing traversal stats on the GPU.\nNote that this option slows down the performance.");

        // Fetch data and show stats if available.
        copyStatsToCPU();
        if (mStatsValid)
        {
            widget.text("Stats:");
            widget.tooltip("All averages are per pixel on screen.\n");

            std::ostringstream oss;
            oss << "Inference path length (avg): " << std::fixed << std::setprecision(3) << mStats.avgInferencePathLength << "\n"
                << "Suffix path length (avg): " << std::fixed << std::setprecision(3) << mStats.avgSuffixPathLength << "\n";

            widget.checkbox("Enable logging", mEnableLogging);
            widget.text(oss.str());

            if (mEnableLogging) logInfo("\n" + oss.str());
        }
    }

    bool NRCPixelStats::getStats(NRCPixelStats::Stats& stats)
    {
        copyStatsToCPU();
        if (!mStatsValid)
        {
            logWarning("NRCPixelStats::getStats() - Stats are not valid. Ignoring.");
            return false;
        }
        stats = mStats;
        return true;
    }

    const Texture::SharedPtr NRCPixelStats::getInferencePathLength() const
    {
        assert(!mRunning);
        return mStatsBuffersValid ? mpStatsInferencePathLength : nullptr;
    }

    const Texture::SharedPtr NRCPixelStats::getSuffixPathLength() const
    {
        assert(!mRunning);
        return mStatsBuffersValid ? mpStatsSuffixPathLength : nullptr;
    }

    void NRCPixelStats::copyStatsToCPU()
    {
        assert(!mRunning);
        if (mWaitingForData)
        {
            // Wait for signal.
            mpFence->syncCpu();
            mWaitingForData = false;

            if (mEnabled)
            {
                // Map the stats buffer.
                const uint4* result = static_cast<const uint4*>(mpReductionResult->map(Buffer::MapType::Read));
                assert(result);

                const uint32_t totalInferencePathLength = result[0].x;
                const uint32_t totalSuffixPathLength = result[1].x;
                const uint32_t numPixels = mFrameDim.x * mFrameDim.y;
                assert(numPixels > 0);

                mStats.avgInferencePathLength = (float)totalInferencePathLength / numPixels;
                mStats.avgSuffixPathLength = (float)totalSuffixPathLength / numPixels *
                    NRC::Parameters::trainingPathStride.x * NRC::Parameters::trainingPathStride.y;

                mpReductionResult->unmap();
                mStatsValid = true;
            }
        }
    }

    pybind11::dict NRCPixelStats::Stats::toPython() const
    {
        pybind11::dict d;

        d["avgInferencePathLength"] = avgInferencePathLength;
        d["avgSuffixPathLength"] = avgSuffixPathLength;

        return d;
    }

    //SCRIPT_BINDING(NRCPixelStats)
    //{
    //    pybind11::class_<NRCPixelStats, NRCPixelStats::SharedPtr> nrcPixelStats(m, "NRCPixelStats");
    //    nrcPixelStats.def_property("enabled", &NRCPixelStats::isEnabled, &NRCPixelStats::setEnabled);
    //    nrcPixelStats.def_property_readonly("stats", [](NRCPixelStats* pPixelStats) {
    //        NRCPixelStats::Stats stats;
    //        pPixelStats->getStats(stats);
    //        return stats.toPython();
    //    });
    //}
}
