#pragma once

#ifndef NRC_MATH_HELPERS
#define NRC_MATH_HELPERS

__device__ float4 operator * (float4 a, float4 b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

__device__ float4 operator + (float4 a, float4 b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

__device__ float4 operator / (float4 a, float4 b) {
    return { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
}

__device__ float3 operator * (float3 a, float3 b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

__device__ float3 operator + (float3 a, float3 b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ float3 operator / (float3 a, float3 b) {
    return { a.x / b.x, a.y / b.y, a.z / b.z };
}

__device__ float3 safe_div(float3 a, float3 b) {
    float3 res = a / b;
    res.x = isinf(res.x) || isnan(res.x) ? 0 : res.x;
    res.y = isinf(res.y) || isnan(res.y) ? 0 : res.y;
    res.z = isinf(res.z) || isnan(res.z) ? 0 : res.z;
    return res;
}

__device__ float4 safe_div(float4 a, float4 b) {
    float4 res = a / b;
    res.x = isinf(res.x) || isnan(res.x) ? 0 : res.x;
    res.y = isinf(res.y) || isnan(res.y) ? 0 : res.y;
    res.z = isinf(res.z) || isnan(res.z) ? 0 : res.z;
    res.w = isinf(res.w) || isnan(res.w) ? 0 : res.w;
    return res;
}

template <typename T = float>
__global__ void check_nans(uint32_t n_elements, T* data) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n_elements) return;
    if (isnan(data[i]) || isinf(data[i])) {
        data[i] = (T)0.f;
    }
}

template <typename T>
// call with 1dim invocation or with linear_kernel
//      num_elements: num bytes of output data.
__global__ void trim_cast(uint32_t num_elements, uint32_t stride_in, uint32_t stride_out, float* __restrict__ data_in, float* __restrict__ data_out) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > num_elements) return;
    uint32_t idx = i % stride_out;
    uint32_t elem = i / stride_out;
    data_out[i] = data_in[elem * stride_in + idx];
}
#endif // !1
