cbuffer CompositeCB {
    bool gVisualizeFactor;
}; 

Texture2D<float4> bias;
Texture2D<float4> factor;
Texture2D<float4> radiance;
/*Texture2D<float4> diffuse;
Texture2D<float4> specular;*/
RWTexture2D<float4> output;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID){
    const uint2 pixel = dispatchThreadId.xy;
    float4 result = bias[pixel];
    // factor == 0 means we don't need to query the network.
    if (any(factor[pixel])) result += factor[pixel] * radiance[pixel];
    output[pixel] = result;
    if (gVisualizeFactor) output[pixel] = factor[pixel];
}
