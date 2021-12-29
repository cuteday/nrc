cbuffer CompositeCB {
    uint gVisualizeMode;
    bool gReflectanceFact;
}; 

Texture2D<float4> bias;
Texture2D<float4> factor;
Texture2D<float4> radiance;
Texture2D<float4> reflectance;
RWTexture2D<float4> output;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID){
    const uint2 pixel = dispatchThreadId.xy;
    float4 result = bias[pixel];
    // float4 reflectance = float4(diffuse[pixel].xyz + specular[pixel].xyz, 1);
    // factor == 0 means we don't need to query the network, these situations include:
    // e.g. last hit on emissive primitives (envmaps, lights, etc.)
    if (any(factor[pixel])) {
        float4 defact = gReflectanceFact ? reflectance[pixel] : float4(1);
        result += factor[pixel] * radiance[pixel] * defact;
    }
    output[pixel] = result;
    if (gVisualizeMode > 1 && any(factor[pixel]))
    {
        if (gVisualizeMode == 2)
            output[pixel] = radiance[pixel];
        else if (gVisualizeMode == 3)
            output[pixel] = factor[pixel];
        else if (gVisualizeMode == 4)
            output[pixel] = bias[pixel];
        else if (gVisualizeMode == 5)
            output[pixel] = reflectance[pixel];
    }
}
