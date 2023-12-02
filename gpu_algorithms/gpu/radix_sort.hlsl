#define GROUP_SIZE_LOG2 6
#define GROUP_SIZE (1 << GROUP_SIZE_LOG2)
#define HW_WAVE_SIZE 32
#define BITS_PER_RADIX 8
#define RADIX_COUNTS 32/(BITS_PER_RADIX)
#define THREAD_DWORD_COMPONENTS (GROUP_SIZE / 32)

#include "thread_utils.hlsl"

cbuffer RadixArgs : register(b0)
{
    int4 g_bufferArgs0;
}

#define g_inputCount g_bufferArgs0.x
#define g_batchCount g_bufferArgs0.y
#define g_radixMask (uint)g_bufferArgs0.z
#define g_radixShift g_bufferArgs0.w

Buffer<uint> g_inputBuffer : register(t0);
RWBuffer<uint> g_outputRadixTable : register(u0);

#define RADIX_TABLE_SIZE RADIX_COUNTS * THREAD_DWORD_COMPONENTS
groupshared uint gs_localRadixTable[RADIX_TABLE_SIZE];
groupshared uint gs_radixCounts[RADIX_COUNTS];

[numthreads(GROUP_SIZE, 1, 1)]
void csCountScatterBuckets(
    int3 dispatchThreadID : SV_DispatchThreadID,
    int groupIndex : SV_GroupIndex)
{
    int batchIndex = dispatchThreadID.x / g_batchCount;
    int batchEnd = min(g_inputCount, batchIndex * g_batchCount + g_batchCount);

    int threadComponentOffset = groupIndex >> DWORD_BIT_SIZE_LOG2; // divide by 32
    int threadComponentBitIndex = groupIndex & (DWORD_BIT_SIZE - 1); // modulus 32
    uint threadPrefixMask[THREAD_DWORD_COMPONENTS];

    int i, k;

    for (i = 0; i < THREAD_DWORD_COMPONENTS; ++i)
        threadPrefixMask[i] = i >= threadComponentOffset ? (i == threadComponentOffset ? ((1u << threadBit) - 1u) : 0) : ~0;

    for (i = groupIndex; i < RADIX_COUNTS; i += GROUP_SIZE)
        gs_radixCounts[i] = 0;

    GroupMemoryBarrierWithGroupSync();

    for (i = dispatchThreadID.x; i < batchEnd; i += GROUP_SIZE)
    {
        for (k = groupIndex; k < RADIX_TABLE_SIZE; k += GROUP_SIZE)
            gs_localRadixTable[k] = 0;

        GroupMemoryBarrierWithGroupSync();

        uint value = i < g_inputCount ? g_inputBuffer[i] : ~0u;
        uint radix = (value >> g_radixShift) & g_radixMask;
        uint unused;
        InterlockOr(gs_localRadixTable[THREAD_DWORD_COMPONENTS*radix + threadComponentOffset], 1u << threadComponentBitIndex, unused);

        GroupMemoryBarrierWithGroupSync();

        uint localOffset = 0, localCount = 0;
        for (k = 0; k < THREAD_DWORD_COMPONENTS; ++k)
        {
            uint radixTableThreadBits = gs_localRadixTable[THREAD_DWORD_COMPONENTS*radix + k]
            localOffset += countbits(radixTableThreadBits & threadPrefixMask[k]);
            localCount += countbits(radixTableThreadBits);
        }

        InterlockAdd(gs_radixCounts[radix], localCount, unused);
    }

    GroupMemoryBarrierWithGroupSync();

    for (i = groupIndex; i < RADIX_COUNTS; i += GROUP_SIZE)
        g_outputRadixTable[batchIndex * RADIX_COUNTS + i] = gs_radixCounts[i];
}
