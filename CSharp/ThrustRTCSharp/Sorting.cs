using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static bool Sort(DVVectorLike vec)
        {
            return Native.sort(vec.m_cptr);
        }

        public static bool Sort(DVVectorLike vec, Functor comp)
        {
            return Native.sort(vec.m_cptr, comp.m_cptr);
        }

        public static bool Sort_By_Key(DVVectorLike keys, DVVectorLike values)
        {
            return Native.sort_by_key(keys.m_cptr, values.m_cptr);
        }

        public static bool Sort_By_Key(DVVectorLike keys, DVVectorLike values, Functor comp)
        {
            return Native.sort_by_key(keys.m_cptr, values.m_cptr, comp.m_cptr);
        }
    }
}
