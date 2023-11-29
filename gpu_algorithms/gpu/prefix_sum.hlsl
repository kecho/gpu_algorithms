// This value must match the group size in prefux_sum.py
#define GROUP_SIZE 128
#define GroupSize GROUP_SIZE
#define HW_WAVE_SIZE 32
#include "thread_utils.hlsl"

Buffer<uint> g_inputBuffer : register(t0);
RWBuffer<uint> g_outputBuffer : register(u0);

cbuffer ConstantsPrefixSum : register(b0)
{
    int4 g_bufferArgs0;
}

#define inputCount g_bufferArgs0.x
#define inputOffset g_bufferArgs0.y
#define outputOffset g_bufferArgs0.z
#define parentOffset g_bufferArgs0.w

[numthreads(GROUP_SIZE, 1, 1)]
void csPrefixSumOnGroup(
    int3 dispatchThreadID : SV_DispatchThreadID,
    int groupIndex : SV_GroupIndex)
{
    int threadID = dispatchThreadID.x;
    uint inputVal = threadID >= inputCount ? 0u : g_inputBuffer[threadID + inputOffset];

    uint outputVal, count;
    ThreadUtils::PrefixExclusive(groupIndex, inputVal, outputVal, count);

#ifndef EXCLUSIVE_PREFIX
    outputVal += inputVal;
#endif
    g_outputBuffer[threadID + outputOffset] = outputVal;
}

[numthreads(GROUP_SIZE, 1, 1)]
void csPrefixSumNextInput(int3 dispatchThreadID : SV_DispatchThreadID, int3 groupID : SV_GroupID)
{
    g_outputBuffer[dispatchThreadID.x] = g_inputBuffer[inputOffset + dispatchThreadID.x * GROUP_SIZE + GROUP_SIZE - 1];
}

groupshared uint g_parentSum;

[numthreads(GROUP_SIZE, 1, 1)]
void csPrefixSumResolveParent(int3 dispatchThreadID : SV_DispatchThreadID, int groupIndex : SV_GroupIndex, int3 groupID : SV_GroupID)
{
    //if (groupIndex == 0)
    //    g_parentSum = groupID.x == 0 ? 0 : g_outputBuffer[parentOffset + groupID.x - 1];

    //no need to do barriers / etc since groupID will trigger a scalar load. We hope!!
    uint parentSum = groupID.x == 0 ? 0 : g_outputBuffer[parentOffset + groupID.x - 1];
    int index = outputOffset + dispatchThreadID.x;
#if EXCLUSIVE_PREFIX
    uint val = g_outputBuffer[index] - g_inputBuffer[index];
    g_outputBuffer[index] = val + parentSum;
#else
    g_outputBuffer[index] += parentSum;
#endif
}
