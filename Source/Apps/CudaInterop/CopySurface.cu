#include "CopySurface.h"
#include <device_launch_parameters.h>

// The CUDA kernel. This sample simply copies the input surface.
template<class T>
__global__ void copySurface(cudaSurfaceObject_t input, cudaSurfaceObject_t output, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        T data;
        surf2Dread(&data, input, sizeof(T) * x, y);
        surf2Dwrite(data, output, sizeof(T) * x, y);
    }
}

// A wrapper function that launches the kernel.
void launchCopySurface(cudaSurfaceObject_t input, cudaSurfaceObject_t output, unsigned int width, unsigned int height, unsigned int format)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    if (format == cudaChannelFormatKindFloat) copySurface<float><<<dimGrid, dimBlock>>>(input, output, width, height);
    else copySurface<int><<<dimGrid, dimBlock>>>(input, output, width, height);
}
