using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static bool Sort(DVVectorLike vec, long begin = 0, long end = -1)
        {
            return Native.sort(vec.m_cptr, (ulong)begin, (ulong)end);
        }

        public static bool Sort(DVVectorLike vec, Functor comp, long begin = 0, long end = -1)
        {
            return Native.sort(vec.m_cptr, comp.m_cptr, (ulong)begin, (ulong)end);
        }

        public static bool Sort_By_Key(DVVectorLike keys, DVVectorLike values, long begin_keys = 0, long end_keys = -1, long begin_values = 0)
        {
            return Native.sort_by_key(keys.m_cptr, values.m_cptr, (ulong)begin_keys, (ulong)end_keys, (ulong)begin_values);
        }

        public static bool Sort_By_Key(DVVectorLike keys, DVVectorLike values, Functor comp, long begin_keys = 0, long end_keys = -1, long begin_values = 0)
        {
            return Native.sort_by_key(keys.m_cptr, values.m_cptr, comp.m_cptr, (ulong)begin_keys, (ulong)end_keys, (ulong)begin_values);
        }
    }
}
