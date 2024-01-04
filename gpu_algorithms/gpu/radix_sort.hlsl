#define BITS_PER_RADIX 8
#define RADIX_COUNTS (1 << BITS_PER_RADIX)

#ifndef GROUP_SIZE
#define GROUP_SIZE 128
#endif

#define HW_WAVE_SIZE 32

#include "thread_utils.hlsl"

#define THREAD_DWORD_COMPONENTS (GROUP_SIZE / DWORD_BIT_SIZE)

#define FLAGS_IS_FIRST_PASS 1 << 0
#define FLAGS_OUTPUT_ORDERING 1 << 1

cbuffer RadixArgs : register(b0)
{
    uint g_inputCount;
    uint g_batchCount;
    uint g_radixMask;
    uint g_radixShift;

    uint g_elementsInBatch;
    uint g_flags;
    uint g_unused1;
    uint g_unused2;
}


Buffer<uint> g_inputBuffer : register(t0);
Buffer<uint> g_inputOrdering : register(t1);
RWBuffer<uint> g_outputBatchOffset : register(u0);
RWBuffer<uint> g_outputRadixTable : register(u1);

#define RADIX_TABLE_SIZE (RADIX_COUNTS * THREAD_DWORD_COMPONENTS)
groupshared uint gs_localRadixTable[RADIX_TABLE_SIZE];
groupshared uint gs_radixCounts[RADIX_COUNTS];

[numthreads(GROUP_SIZE, 1, 1)]
void csCountScatterBuckets(
    int groupIndex : SV_GroupIndex,
    int3 groupID : SV_GroupID)
{
    int batchIndex = groupID.x;
    int batchBegin = batchIndex * g_elementsInBatch;
    int batchEnd = min(g_inputCount, batchBegin + g_elementsInBatch);

    int threadComponentOffset = groupIndex >> DWORD_BIT_SIZE_LOG2; // divide by 32
    int threadComponentBitIndex = groupIndex & (DWORD_BIT_SIZE - 1); // modulus 32
    uint threadPrefixMask[THREAD_DWORD_COMPONENTS];

    int bi, k, unused;

    for (k = 0; k < THREAD_DWORD_COMPONENTS; ++k)
        threadPrefixMask[k] = k >= threadComponentOffset ? (k == threadComponentOffset ? ((1u << threadComponentBitIndex) - 1u) : 0) : ~0;

    for (k = groupIndex; k < RADIX_COUNTS; k += GROUP_SIZE)
        gs_radixCounts[k] = 0;

    GroupMemoryBarrierWithGroupSync();
    
    uint batchIterations = (batchEnd - batchBegin + GROUP_SIZE - 1)/GROUP_SIZE;

    bool outputsOrdering = (g_flags & FLAGS_OUTPUT_ORDERING) != 0;
    bool isFirstPass = (g_flags & FLAGS_IS_FIRST_PASS) != 0;
    bool sampleOrdering = outputsOrdering && !isFirstPass;

    [loop]
    for (bi = 0; bi < batchIterations; ++bi)
    {
        uint inputOffset = batchBegin + bi * GROUP_SIZE + groupIndex;

        [loop]
        for (k = groupIndex; k < RADIX_TABLE_SIZE; k += GROUP_SIZE)
            gs_localRadixTable[k] = 0;

        // wait for writes in gs_localRadixTable
        GroupMemoryBarrierWithGroupSync();

        uint i = inputOffset; 
        if (sampleOrdering)
            i = inputOffset < g_inputCount ? g_inputOrdering[inputOffset] : ~0u;

        uint value = i < g_inputCount ? g_inputBuffer[i] : ~0u;
        uint radix = (value >> g_radixShift) & g_radixMask;
        InterlockedOr(gs_localRadixTable[THREAD_DWORD_COMPONENTS*radix + threadComponentOffset], 1u << threadComponentBitIndex, unused);

        // wait atomics in gs_localRadixTable
        GroupMemoryBarrierWithGroupSync();

        uint localOffset = 0;

        [unroll(THREAD_DWORD_COMPONENTS)]
        for (k = 0; k < THREAD_DWORD_COMPONENTS; ++k)
            localOffset += countbits(gs_localRadixTable[THREAD_DWORD_COMPONENTS*radix + k] & threadPrefixMask[k]);

        if (i < g_inputCount)
            g_outputBatchOffset[i] = localOffset + gs_radixCounts[radix];

        // wait for reads in gs_radixCounts (we are about to write to it)
        GroupMemoryBarrierWithGroupSync();

        for (uint usedRadix = groupIndex; usedRadix < RADIX_COUNTS; usedRadix += GROUP_SIZE)
        {
            uint localCountForRadix = 0;

            [unroll(THREAD_DWORD_COMPONENTS)]
            for (k = 0; k < THREAD_DWORD_COMPONENTS; ++k)
                localCountForRadix += countbits(gs_localRadixTable[THREAD_DWORD_COMPONENTS*usedRadix + k]);

            gs_radixCounts[usedRadix] += localCountForRadix;
        }

        // wait for reads in gs_localRadixTable. We are about to write to it in the next iteration.
        GroupMemoryBarrierWithGroupSync();
    }

    // No need for this barrier!, since the loop above has one.
    //GroupMemoryBarrierWithGroupSync();

    [loop]
    for (k = groupIndex; k < RADIX_COUNTS; k += GROUP_SIZE)
        g_outputRadixTable[batchIndex * RADIX_COUNTS + k] = gs_radixCounts[k];
}

Buffer<uint> g_inputCounterTable : register(t0);
RWBuffer<uint> g_outputCounterTablePrefix : register(u0);
RWBuffer<uint> g_outputRadixTotalCounts : register(u1);
        
[numthreads(GROUP_SIZE, 1, 1)]
void csPrefixCountTable(
    int groupIndex : SV_GroupIndex,
    int3 groupID : SV_GroupID)
{
    uint radix = groupID.x;
    uint tb = 0;
    uint radixCounts = 0;
    uint threadBatchesCount = (g_batchCount + GROUP_SIZE - 1)/ GROUP_SIZE;
    for (tb = 0; tb < threadBatchesCount; ++tb)
    {
        uint i = tb * GROUP_SIZE + groupIndex;
    
        uint countValue = i < g_batchCount ? g_inputCounterTable[i * RADIX_COUNTS + radix] : 0;

        uint batchOffset, batchCount;
        ThreadUtils::PrefixExclusive(groupIndex, countValue, batchOffset, batchCount);

        // Mandatory barrier: the prefix above could be using LDS, so the next iteration of this 
        // loop could cause read / write collisions.
        GroupMemoryBarrierWithGroupSync();
    
        if (i < g_batchCount)
            g_outputCounterTablePrefix[radix * g_batchCount + i] = batchOffset + radixCounts;

        radixCounts += batchCount;
    }

    if (groupIndex == 0)
        g_outputRadixTotalCounts[radix] = radixCounts;
}

Buffer<uint> g_inputRadixTotalCounts : register(t0);
RWBuffer<uint> g_outputGlobalPrefix : register(u0);

[numthreads(GROUP_SIZE, 1, 1)]
void csPrefixGlobalTable(int groupIndex : SV_GroupIndex)
{
    uint radixVal = g_inputRadixTotalCounts[groupIndex];
    uint offset, unused;
    ThreadUtils::PrefixExclusive(groupIndex, radixVal, offset, unused);
    g_outputGlobalPrefix[groupIndex] = offset;
}

Buffer <uint> g_inputUnsorted : register(t0);
Buffer <uint> g_inputUnsortedOrdering : register(t1);
Buffer <uint> g_inputLocalBatchOffset : register(t2);
Buffer <uint> g_inputCounterTablePrefix : register(t3);
Buffer <uint> g_inputGlobalPrefix : register(t4);
RWBuffer<uint> g_outputSorted : register(u0);

[numthreads(GROUP_SIZE, 1, 1)]
void csScatterOutput(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint groupIndex : SV_GroupIndex,
    uint3 groupID : SV_GroupID)
{
    uint batchIndex = groupID.x;
    uint batchOffset = groupIndex;
    uint i = dispatchThreadID.x;
    bool outputsOrdering = g_flags & FLAGS_OUTPUT_ORDERING;
    if (outputsOrdering && (g_flags & FLAGS_IS_FIRST_PASS) == 0)
        i = dispatchThreadID.x < g_inputCount ? g_inputUnsortedOrdering[dispatchThreadID.x] : ~0u;

    uint value = i < g_inputCount ? g_inputUnsorted[i] : ~0;
    uint radix = (value >> g_radixShift) & g_radixMask;
    if (i < g_inputCount)
    {
        uint outputIndex = g_inputGlobalPrefix[radix] + g_inputCounterTablePrefix[radix * g_batchCount + batchIndex] + g_inputLocalBatchOffset[i];
        g_outputSorted[outputIndex] = outputsOrdering ? i : value;
    }
}
