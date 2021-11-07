#pragma once
#include "Core/Framework.h"
#include "Core/API/Texture.h"
#include "Core/API/RenderContext.h"

#include <cuda_runtime.h>

namespace FalcorCUDA
{
    /** Initializes the CUDA driver API. Returns true if successful, false otherwise.
    */
    bool initCUDA();

    /** Imports the texture into a CUDA mipmapped array and returns the array in mipmappedArray. This method should only be called once per
        texture resource.
        \param pTex Pointer to the texture being imported
        \param mipmappedArray Reference to the array to import to
        \param usageFlags The requested flags to be bound to the mipmapped array
        \return True if successful, false otherwise
    */
    bool importTextureToMipmappedArray(Falcor::Texture::SharedPtr pTex, cudaMipmappedArray_t& mipmappedArray, uint32_t cudaUsageFlags);

    /** Maps a texture to a surface object which can be read and written within a CUDA kernel.
        This method should only be called once per texture on initial load. Store the returned surface object for repeated use.
        \param pTex Pointer to the texture being mapped
        \param usageFlags The requested flags to be bound to the underlying mipmapped array that will be used to create the surface object
        \return The surface object that the input texture is bound to
    */
    cudaSurfaceObject_t mapTextureToSurface(Falcor::Texture::SharedPtr pTex, uint32_t usageFlags);
};
