#pragma once
#include <cuda_runtime.h>

#include <Falcor.h>


namespace FalcorCUDA
{
    // Initializes the CUDA driver API. Returns true if successful, false otherwise.
    bool initCUDA();

    cudaExternalMemory_t importExternalMemory(Falcor::Resource::SharedPtr pResource);
    void* mapExternalMemory(const cudaExternalMemory_t& externalMemory, unsigned long long size);
    
    void* importResourceToDevicePointer(Falcor::Resource::SharedPtr pResource);
    cudaMipmappedArray_t importTextureToMipmappedArray(Falcor::Texture::SharedPtr pTexture, uint32_t cudaUsageFlags);
};
