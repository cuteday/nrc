#include "FalcorCUDA.h"
#include <cuda.h>
#include <AccCtrl.h>
#include <aclapi.h>
#include "Core/API/Device.h"

#define CU_CHECK_SUCCESS(x)                                                         \
    do {                                                                            \
        CUresult result = x;                                                        \
        if (result != CUDA_SUCCESS)                                                 \
        {                                                                           \
            const char* msg;                                                        \
            cuGetErrorName(result, &msg);                                           \
            logError("CUDA Error: " #x " failed with error " + std::string(msg));   \
            return 0;                                                               \
        }                                                                           \
    } while(0)

#define CUDA_CHECK_SUCCESS(x)                                                                            \
    do {                                                                                                 \
        cudaError_t result = x;                                                                          \
        if (result != cudaSuccess)                                                                       \
        {                                                                                                \
            logError("CUDA Error: " #x " failed with error " + std::string(cudaGetErrorString(result))); \
            return 0;                                                                                    \
        }                                                                                                \
    } while(0) 


using namespace Falcor;

namespace
{
    uint32_t gNodeMask;
    CUdevice  gCudaDevice;
    CUcontext gCudaContext;
    CUstream  gCudaStream;
}

namespace FalcorCUDA
{
    bool initCUDA()
    {
        CU_CHECK_SUCCESS(cuInit(0));
        int32_t firstGPUID = -1;
        cudaDeviceProp prop;
        int32_t count;
        cudaError_t err = cudaGetDeviceCount(&count);

        for (int32_t i = 0; i < count; ++i)
        {
            err = cudaGetDeviceProperties(&prop, i);
            if (prop.major >= 3)
            {
                firstGPUID = i;
                break;
            }
        }

        if (firstGPUID < 0)
        {
            logError("No CUDA 10 compatible GPU found");
            return false;
        }
        gNodeMask = prop.luidDeviceNodeMask;
        CUDA_CHECK_SUCCESS(cudaSetDevice(firstGPUID));
        CU_CHECK_SUCCESS(cuDeviceGet(&gCudaDevice, firstGPUID));
        CU_CHECK_SUCCESS(cuCtxCreate(&gCudaContext, 0, gCudaDevice));
        CU_CHECK_SUCCESS(cuStreamCreate(&gCudaStream, CU_STREAM_DEFAULT));
        return true;
    }

    cudaExternalMemory_t test_import_buffer_to_cuda(Falcor::Buffer::SharedPtr pBuffer) {
        HANDLE sharedHandle = pBuffer->getSharedApiHandle();
        if (sharedHandle == NULL) {
            logError("CUDA::importing buffer failed");
        }

        cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
        externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
        externalMemoryHandleDesc.size = pBuffer->getSize();
        externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

        cudaExternalMemory_t externalMemory;
        CUDA_CHECK_SUCCESS(cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));

        return externalMemory;
    }

    void* mapBufferOntoExternalMemory(const cudaExternalMemory_t& externalMemory, unsigned long long size) {
        void* ptr = nullptr;
        cudaExternalMemoryBufferDesc desc = {};

        desc.offset = 0;
        desc.size = size;

        CUDA_CHECK_SUCCESS(cudaExternalMemoryGetMappedBuffer(&ptr, externalMemory, &desc));
        return ptr;
    }
}
