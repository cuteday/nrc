#pragma once
#include <cuda_runtime.h>

#include <Falcor.h>


namespace FalcorCUDA
{
    /** Initializes the CUDA driver API. Returns true if successful, false otherwise.
    */
    bool initCUDA();

    cudaExternalMemory_t test_import_buffer_to_cuda(Falcor::Buffer::SharedPtr pBuffer);

    void* mapBufferOntoExternalMemory(const cudaExternalMemory_t& externalMemory, unsigned long long size);
};
