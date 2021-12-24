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
__global__ void chkNaN(uint32_t n_elements, T* data) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n_elements) return;
    if (isnan(data[i]) || isinf(data[i])) {
        data[i] = (T)0.f;
    }
}


#endif // !1
