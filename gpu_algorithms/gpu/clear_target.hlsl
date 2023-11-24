RWTexture2D<float4> g_output : register(u0);
cbuffer Constants : register(b0)
{
    float4 clearColor;
}

[numthreads(8,8,1)]
void csMainClear(int2 dti : SV_DispatchThreadID)
{
    g_output[dti] = clearColor;
}

cbuffer ConstantsUintBuff : register(b0)
{
    uint g_uintClearVal;
    int g_clearOffset;
    int g_clearValSize;
}

RWBuffer<uint> g_output_buff_uint : register(u0);
[numthreads(64,1,1)]
void csMainClearUintBuffer(int3 dti : SV_DispatchThreadID)
{
    if (dti.x >= g_clearValSize)
        return;

    g_output_buff_uint[g_clearOffset + dti.x] = g_uintClearVal;
}
