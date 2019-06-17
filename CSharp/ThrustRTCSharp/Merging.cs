using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static bool Merge(DVVectorLike vec1, DVVectorLike vec2, DVVectorLike vec_out, long begin1 = 0, long end1 = -1, long begin2 = 0, long end2 = -1, long begin_out = 0)
        {
            return Native.merge(vec1.m_cptr, vec2.m_cptr, vec_out.m_cptr, (ulong)begin1, (ulong)end1, (ulong)begin2, (ulong)end2, (ulong)begin_out);
        }

        public static bool Merge(DVVectorLike vec1, DVVectorLike vec2, DVVectorLike vec_out, Functor comp, long begin1 = 0, long end1 = -1, long begin2 = 0, long end2 = -1, long begin_out = 0)
        {
            return Native.merge(vec1.m_cptr, vec2.m_cptr, vec_out.m_cptr, comp.m_cptr, (ulong)begin1, (ulong)end1, (ulong)begin2, (ulong)end2, (ulong)begin_out);
        }

        public static bool Merge_By_Key(DVVectorLike keys1, DVVectorLike keys2, DVVectorLike value1, DVVectorLike value2, DVVectorLike keys_out, DVVectorLike value_out, long begin_keys1 = 0, long end_keys1 = -1, long begin_keys2 = 0, long end_keys2 = -1, long begin_value1 = 0, long begin_value2 = 0, long begin_keys_out = 0, long begin_value_out = 0)
        {
            return Native.merge_by_key(keys1.m_cptr, keys2.m_cptr, value1.m_cptr, value2.m_cptr, keys_out.m_cptr, value_out.m_cptr, (ulong)begin_keys1, (ulong)end_keys1, (ulong)begin_keys2, (ulong)end_keys2, (ulong)begin_value1, (ulong)begin_value2, (ulong)begin_keys_out, (ulong)begin_value_out);
        }

        public static bool Merge_By_Key(DVVectorLike keys1, DVVectorLike keys2, DVVectorLike value1, DVVectorLike value2, DVVectorLike keys_out, DVVectorLike value_out, Functor comp, long begin_keys1 = 0, long end_keys1 = -1, long begin_keys2 = 0, long end_keys2 = -1, long begin_value1 = 0, long begin_value2 = 0, long begin_keys_out = 0, long begin_value_out = 0)
        {
            return Native.merge_by_key(keys1.m_cptr, keys2.m_cptr, value1.m_cptr, value2.m_cptr, keys_out.m_cptr, value_out.m_cptr, comp.m_cptr, (ulong)begin_keys1, (ulong)end_keys1, (ulong)begin_keys2, (ulong)end_keys2, (ulong)begin_value1, (ulong)begin_value2, (ulong)begin_keys_out, (ulong)begin_value_out);
        }
    }
}
