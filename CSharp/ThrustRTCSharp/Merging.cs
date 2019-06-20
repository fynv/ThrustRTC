using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static bool Merge(DVVectorLike vec1, DVVectorLike vec2, DVVectorLike vec_out)
        {
            return Native.merge(vec1.m_cptr, vec2.m_cptr, vec_out.m_cptr);
        }

        public static bool Merge(DVVectorLike vec1, DVVectorLike vec2, DVVectorLike vec_out, Functor comp)
        {
            return Native.merge(vec1.m_cptr, vec2.m_cptr, vec_out.m_cptr, comp.m_cptr);
        }

        public static bool Merge_By_Key(DVVectorLike keys1, DVVectorLike keys2, DVVectorLike value1, DVVectorLike value2, DVVectorLike keys_out, DVVectorLike value_out)
        {
            return Native.merge_by_key(keys1.m_cptr, keys2.m_cptr, value1.m_cptr, value2.m_cptr, keys_out.m_cptr, value_out.m_cptr);
        }

        public static bool Merge_By_Key(DVVectorLike keys1, DVVectorLike keys2, DVVectorLike value1, DVVectorLike value2, DVVectorLike keys_out, DVVectorLike value_out, Functor comp)
        {
            return Native.merge_by_key(keys1.m_cptr, keys2.m_cptr, value1.m_cptr, value2.m_cptr, keys_out.m_cptr, value_out.m_cptr, comp.m_cptr);
        }
    }
}
