#pragma once
#include <cuda_runtime.h>

#include <Falcor.h>
#include "FalcorCUDA.h"

using namespace Falcor;

namespace NRC {

    struct Sample {
        Falcor::uint val;
    };

    class NRCInterface {

    public:
        using SharedPtr = std::shared_ptr<NRCInterface>;
        using WeakPtr = std::weak_ptr<NRCInterface>;
        using SharedConstPtr = std::shared_ptr<const NRCInterface>;

        NRCInterface();

        static void test_buffer_cuda_interop(Falcor::Buffer::SharedPtr pBuffer) {
            cudaExternalMemory_t externalMemory;
            externalMemory = FalcorCUDA::test_import_buffer_to_cuda(pBuffer);
            Sample* samples_d = (Sample*)FalcorCUDA::mapBufferOntoExternalMemory(externalMemory, pBuffer->getSize());
            //Buffer::SharedPtr countBuffer = pBuffer->getUAVCounter();
            //uint count = *(uint* )countBuffer->map(Buffer::MapType::Read);
            //countBuffer->unmap();
            //Sample* samples_h = (Sample*)malloc(2000 * sizeof(NRC::Sample));
            //cudaMemcpy(samples_h, samples_d, 2000 * sizeof(NRC::Sample), cudaMemcpyDeviceToHost);
            //logInfo("We have " + std::to_string(count) + "samples!");
            //for (uint i = 0; i < count; i++) {
            //    logInfo("Logging samples : " + std::to_string(samples_h[i].val));
            //}
            //free(samples_h);
            cudaFree(samples_d);
        }

        static void test_texture_cuda_interop(Falcor::Texture::SharedPtr pTex) {
        }

        static void test_tcnn_interop() {

        }

    private:

    };
}
