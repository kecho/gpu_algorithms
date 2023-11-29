#ifndef __THREAD_UTILS__
#define __THREAD_UTILS__

#ifndef GROUP_SIZE
    #error "ThreadUtils.hlsl requires definition of GROUP_SIZE"
#endif

#define DWORD_BIT_SIZE_LOG2 5
#define DWORD_BIT_SIZE (1 << DWORD_BIT_SIZE_LOG2)

#define THREAD_UTILS_MODE_IMPL_LDS 0
#define THREAD_UTILS_MODE_GROUPWAVE 1
#define THREAD_UTILS_MODE_SUBWAVE_IN_GROUP 2

#if defined(HW_WAVE_SIZE)
    #if (GROUP_SIZE & (HW_WAVE_SIZE - 1)) != 0
        #error "Group size must be a multiple of the wave size for this library. Current setup not supported."
    #elif (HW_WAVE_SIZE > GROUP_SIZE)
        #error "Group size must be less than the wave size for this library. Current setup not supported."
    #elif HW_WAVE_SIZE == GROUP_SIZE
        #define THREAD_UTILS_MODE THREAD_UTILS_MODE_GROUPWAVE
    #elif HW_WAVE_SIZE < GROUP_SIZE
        #define THREAD_UTILS_MODE THREAD_UTILS_MODE_SUBWAVE_IN_GROUP
    #else
        #error "Unsupported group size / wave size configuration."
    #endif
#else
    #define THREAD_UTILS_MODE THREAD_UTILS_MODE_IMPL_LDS
#endif

#ifndef THREAD_UTILS_MODE 
    #error "THREAD_UTILS_MODE must be defined at this point in this library."
#endif

namespace ThreadUtils
{
#define BIT_MASK_SIZE ((GROUP_SIZE + DWORD_BIT_SIZE - 1)/ DWORD_BIT_SIZE)

struct GroupData2
{
    uint bitMask0[BIT_MASK_SIZE];
    uint bitMask1[BIT_MASK_SIZE];
};

void PrefixBitSum(uint groupThreadIndex, bool bitValue,  out uint  outOffset,  out uint outCount);
void PrefixBit2Sum(uint groupThreadIndex, bool2 bitValues, out uint2 outOffsets, out uint2 outCounts, out GroupData2 groupData);
void PrefixBit2Sum(uint groupThreadIndex, bool2 bitValues, out uint2 outOffsets, out uint2 outCounts)
{
    GroupData2 unused;
    PrefixBit2Sum(groupThreadIndex, bitValues, outOffsets, outCounts, unused);
}
void PrefixExclusive(uint groupThreadIndex, uint value, out uint outOffset, out uint outCount);
void PrefixInclusive(uint groupThreadIndex, uint value, out uint outOffset, out uint outCount)
{
    PrefixExclusive(groupThreadIndex, value, outOffset, outCount);
    outOffset += value;
}
uint CalculateGlobalStorageOffset(RWByteAddressBuffer counterBuffer, uint groupThreadIndex, bool bitValue);
uint CalculateGlobalValueStorageOffset(RWByteAddressBuffer counterBuffer, uint groupThreadIndex, uint valueCount);

#if THREAD_UTILS_MODE == THREAD_UTILS_MODE_IMPL_LDS
                
groupshared uint gs_BitMask0[BIT_MASK_SIZE];
groupshared uint gs_BitMask1[BIT_MASK_SIZE];

void PrefixBitSum(uint groupThreadIndex, bool bitValue,  out uint  outOffset,  out uint outCount)
{
    if (groupThreadIndex < BIT_MASK_SIZE)
        gs_BitMask0[groupThreadIndex] = 0;

    GroupMemoryBarrierWithGroupSync();

    uint maskOffset = groupThreadIndex >> DWORD_BIT_SIZE_LOG2;
    uint maskBit = (groupThreadIndex & (DWORD_BIT_SIZE - 1));
    uint mask = 1u << maskBit;
    uint unused; 

    [branch]
    if (bitValue)
        InterlockedOr(gs_BitMask0[maskOffset], mask, unused);

    GroupMemoryBarrierWithGroupSync();

    outOffset = 0;
    outCount = 0;
    {
        [unroll]
        for (uint i = 0; i < BIT_MASK_SIZE; ++i)
        {
            uint maskCount = countbits(gs_BitMask0[i]);
            if (i < maskOffset)
                outOffset += maskCount;
            outCount += maskCount;
        }
        uint v = gs_BitMask0[maskOffset];
        outOffset += countbits((mask - 1u) & v);
    }
}

void PrefixBit2Sum(uint groupThreadIndex,bool2 bitValues,out uint2 outOffsets, out uint2 outCounts, out GroupData2 groupData)
{
    if (groupThreadIndex < BIT_MASK_SIZE)
    {
        gs_BitMask0[groupThreadIndex] = 0;
        gs_BitMask1[groupThreadIndex] = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    uint maskOffset = groupThreadIndex >> DWORD_BIT_SIZE_LOG2;
    uint maskBit = (groupThreadIndex & (DWORD_BIT_SIZE - 1));
    uint mask = 1u << maskBit;
    uint unused;
    [branch]
    if (bitValues.x)
        InterlockedOr(gs_BitMask0[maskOffset], mask, unused);
    if (bitValues.y)
        InterlockedOr(gs_BitMask1[maskOffset], mask, unused);

    GroupMemoryBarrierWithGroupSync();

    outOffsets = 0;
    outCounts = 0;
    {
        [unroll(BIT_MASK_SIZE)]
        for (uint i = 0; i < BIT_MASK_SIZE; ++i)
        {
            uint2 maskCounts = uint2(countbits(gs_BitMask0[i]), countbits(gs_BitMask1[i]));
            if (i < maskOffset)
                outOffsets += maskCounts;
            outCounts += maskCounts;
        }
        uint2 v = uint2(gs_BitMask0[maskOffset],gs_BitMask1[maskOffset]);
        outOffsets += uint2(countbits((mask - 1u) & v.x), countbits((mask - 1u) & v.y));
    }

    [unroll(BIT_MASK_SIZE)]
    for (uint i = 0; i < BIT_MASK_SIZE; ++i)
    {
        groupData.bitMask0[i] = gs_BitMask0[i];
        groupData.bitMask1[i] = gs_BitMask1[i];
    }
}

groupshared uint gs_PrefixCache[GROUP_SIZE];

void PrefixExclusive(uint groupThreadIndex, uint value, out uint outOffset, out uint outCount)
{
    gs_PrefixCache[groupThreadIndex] = value;

    GroupMemoryBarrierWithGroupSync();

    for (uint i = 1; i < GROUP_SIZE; i <<= 1)
    {
        uint sampleVal = groupThreadIndex >= i ? gs_PrefixCache[groupThreadIndex - i] : 0u;

        GroupMemoryBarrierWithGroupSync();

        gs_PrefixCache[groupThreadIndex] += sampleVal;

        GroupMemoryBarrierWithGroupSync();
    }

    outOffset = gs_PrefixCache[groupThreadIndex] - value;
    outCount = gs_PrefixCache[GROUP_SIZE - 1];
}

uint CalculateGlobalStorageOffset(RWByteAddressBuffer counterBuffer, uint groupThreadIndex, bool bitValue)
{
    uint localOffset, totalCount;
    PrefixBitSum(groupThreadIndex, bitValue, localOffset, totalCount);

    if (groupThreadIndex == 0 && totalCount > 0)
    {
        uint globalOffset = 0;
        counterBuffer.InterlockedAdd(0, totalCount, globalOffset);
        gs_BitMask0[0] = globalOffset;
    }

    GroupMemoryBarrierWithGroupSync();

    return gs_BitMask0[0] + localOffset;
}

uint CalculateGlobalValueStorageOffset(RWByteAddressBuffer counterBuffer, uint groupThreadIndex, uint valueCount)
{
    uint localOffset, totalCount;
    PrefixExclusive(groupThreadIndex, valueCount, localOffset, totalCount);

    if (groupThreadIndex == 0 && totalCount > 0)
    {
        uint globalOffset = 0;
        counterBuffer.InterlockedAdd(0, totalCount, globalOffset);
        gs_PrefixCache[0] = globalOffset;
    }

    GroupMemoryBarrierWithGroupSync();

    return gs_PrefixCache[0] + localOffset;
}

#elif THREAD_UTILS_MODE == THREAD_UTILS_MODE_GROUPWAVE || THREAD_UTILS_MODE == THREAD_UTILS_MODE_SUBWAVE_IN_GROUP

#if THREAD_UTILS_MODE == THREAD_UTILS_MODE_SUBWAVE_IN_GROUP
#define GROUP_WAVE_COMPONENT_SIZE (GROUP_SIZE / HW_WAVE_SIZE)
groupshared uint gs_GroupCache0[GROUP_WAVE_COMPONENT_SIZE];
groupshared uint gs_GroupCache1[GROUP_WAVE_COMPONENT_SIZE];
#endif //THREAD_UTILS_MODE == THREAD_UTILS_MODE_SUBWAVE_IN_GROUP

void PrefixBitSum(uint groupThreadIndex, bool bitValue, out uint outOffset, out uint outCount)
{
    uint widx = WaveReadLaneFirst(groupThreadIndex / HW_WAVE_SIZE);
    uint prefixOffset = WavePrefixCountBits(bitValue);
    uint waveCount = WaveActiveCountBits(bitValue);

#if THREAD_UTILS_MODE == THREAD_UTILS_MODE_SUBWAVE_IN_GROUP
    if (WaveIsFirstLane())
        gs_GroupCache0[widx] = waveCount;

    GroupMemoryBarrierWithGroupSync();

    outCount = 0;
    outOffset = prefixOffset;
    for (uint pwid = 0; pwid < GROUP_WAVE_COMPONENT_SIZE; ++pwid)
    {
        uint cacheVal = gs_GroupCache0[pwid];
        if (pwid < widx) 
            outOffset += cacheVal;
        outCount += cacheVal;
    }
#else
    outOffset = prefixOffset;
    outCount = waveCount;
#endif
}

void PrefixBit2Sum(uint groupThreadIndex, bool2 bitValues, out uint2 outOffsets, out uint2 outCounts, out GroupData2 groupData)
{ 
    uint widx = WaveReadLaneFirst(groupThreadIndex / HW_WAVE_SIZE);
    uint2 prefixOffsets = uint2(WavePrefixCountBits(bitValues.x), WavePrefixCountBits(bitValues.y));
    uint2 waveCounts = uint2(WaveActiveCountBits(bitValues.x), WaveActiveCountBits(bitValues.y));

#if THREAD_UTILS_MODE == THREAD_UTILS_MODE_SUBWAVE_IN_GROUP
    if (WaveIsFirstLane())
    {
        gs_GroupCache0[widx] = waveCounts.x;
        gs_GroupCache1[widx] = waveCounts.y;
    }

    GroupMemoryBarrierWithGroupSync();

    outCounts = 0;
    outOffsets = prefixOffsets;
    for (uint pwid = 0; pwid < GROUP_WAVE_COMPONENT_SIZE; ++pwid)
    {
        uint2 cacheVals = uint2(gs_GroupCache0[pwid], gs_GroupCache1[pwid]);
        if (pwid < widx) 
            outOffsets += cacheVals;
        outCounts += cacheVals;
    }
#else
    outOffsets = prefixOffsets;
    outCounts = waveCounts;
#endif

    {
        const uint groupOffset = ((groupThreadIndex + DWORD_BIT_SIZE - 1) / DWORD_BIT_SIZE);
        [unroll(BIT_MASK_SIZE)]
        for (int i = 0; i < BIT_MASK_SIZE; i++)
        {
            const uint ballotIndex = i % 2;
            groupData.bitMask0[i + groupOffset] = WaveActiveBallot(bitValues.x)[ballotIndex];
            groupData.bitMask1[i + groupOffset] = WaveActiveBallot(bitValues.y)[ballotIndex];
        }
    }
}

void PrefixExclusive(uint groupThreadIndex, uint value, out uint outOffset, out uint outCount)
{
    uint widx = WaveReadLaneFirst(groupThreadIndex / HW_WAVE_SIZE);
    uint prefixOffset = WavePrefixSum(value);
    uint waveCount = WaveActiveSum(value);

#if THREAD_UTILS_MODE == THREAD_UTILS_MODE_SUBWAVE_IN_GROUP
    if (WaveIsFirstLane())
        gs_GroupCache0[widx] = waveCount;

    GroupMemoryBarrierWithGroupSync();

    outCount = 0;
    outOffset = prefixOffset;

    for (uint pwid = 0; pwid < GROUP_WAVE_COMPONENT_SIZE; ++pwid)
    {
        uint cacheVal = gs_GroupCache0[pwid];
        if (pwid < widx)
            outOffset += cacheVal;
        outCount += cacheVal;
    }
#else
    outOffset = prefixOffset;
    outCount = waveCount;
#endif
}

uint CalculateGlobalStorageOffset(RWByteAddressBuffer counterBuffer, uint groupThreadIndex, bool bitValue)
{
    uint localOffset, totalCount;
    PrefixBitSum(groupThreadIndex, bitValue, localOffset, totalCount);

#if THREAD_UTILS_MODE == THREAD_UTILS_MODE_SUBWAVE_IN_GROUP
    if (groupThreadIndex == 0 && totalCount > 0)
    {
        uint globalOffset = 0;
        counterBuffer.InterlockedAdd(0, totalCount, globalOffset);
        gs_GroupCache0[0] = globalOffset;
    }

    GroupMemoryBarrierWithGroupSync();

    return gs_GroupCache0[0] + localOffset;
#else
    uint globalOffset = 0;
    if (WaveIsFirstLane() && totalCount > 0)
        counterBuffer.InterlockedAdd(0, totalCount, globalOffset);

    return WaveReadLaneFirst(globalOffset) + localOffset;
#endif
}

uint CalculateGlobalValueStorageOffset(RWByteAddressBuffer counterBuffer, uint groupThreadIndex, uint valueCount)
{
    uint localOffset, totalCount;
    PrefixExclusive(groupThreadIndex, valueCount, localOffset, totalCount);

#if THREAD_UTILS_MODE == THREAD_UTILS_MODE_SUBWAVE_IN_GROUP
    if (groupThreadIndex == 0 && totalCount > 0)
    {
        uint globalOffset = 0;
        counterBuffer.InterlockedAdd(0, totalCount, globalOffset);
        gs_GroupCache0[0] = globalOffset;
    }

    GroupMemoryBarrierWithGroupSync();

    return gs_GroupCache0[0] + localOffset;
#else
    uint globalOffset = 0;
    if (WaveIsFirstLane() && totalCount > 0)
        counterBuffer.InterlockedAdd(0, totalCount, globalOffset);

    return WaveReadLaneFirst(globalOffset) + localOffset;
#endif
}

#else //THREAD_UTILS_MODE unknown
    #error "Implementation for thread utilities not supported."
#endif

}

#endif
