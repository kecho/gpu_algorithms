#ifndef __THREADING_H__
#define __THREADING_H__

#ifndef GroupSize
#error "Must define a group size"
#endif

namespace Threading
{

groupshared uint gs_groupCache[GroupSize];

struct Group
{
    uint m_threadID;
    void init(uint groupThreadIndex)
    {
        m_threadID = groupThreadIndex;
    }

    void prefixExclusive(uint value, out uint sum, out uint count)
    {
        gs_groupCache[m_threadID] = value;

        GroupMemoryBarrierWithGroupSync();

        for (uint i = 1; i < GroupSize; i <<= 1)
        {
            uint prevValue = m_threadID >= i ? gs_groupCache[m_threadID - i] : 0;

            GroupMemoryBarrierWithGroupSync();
    
            gs_groupCache[m_threadID] += prevValue;

            GroupMemoryBarrierWithGroupSync();
        }

        GroupMemoryBarrierWithGroupSync();

        sum = gs_groupCache[m_threadID] - value;
        count = gs_groupCache[GroupSize - 1]; 
    }
};


}

#endif
