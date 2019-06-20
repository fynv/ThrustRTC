using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static bool Gather(DVVectorLike vec_map, DVVectorLike vec_in, DVVectorLike vec_out)
        {
            return Native.gather(vec_map.m_cptr, vec_in.m_cptr, vec_out.m_cptr);
        }

        public static bool Gather_If(DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_in, DVVectorLike vec_out)
        {
            return Native.gather_if(vec_map.m_cptr, vec_stencil.m_cptr, vec_in.m_cptr, vec_out.m_cptr);
        }

        public static bool Gather_If(DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_in, DVVectorLike vec_out, Functor pred)
        {
            return Native.gather_if(vec_map.m_cptr, vec_stencil.m_cptr, vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr);
        }

        public static bool Scatter(DVVectorLike vec_in, DVVectorLike vec_map, DVVectorLike vec_out)
        {
            return Native.scatter(vec_in.m_cptr, vec_map.m_cptr, vec_out.m_cptr);
        }

        public static bool Scatter_If(DVVectorLike vec_in, DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_out)
        {
            return Native.scatter_if(vec_in.m_cptr, vec_map.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr);
        }

        public static bool Scatter_If(DVVectorLike vec_in, DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_out, Functor pred)
        {
            return Native.scatter_if(vec_in.m_cptr, vec_map.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, pred.m_cptr);
        }

        public static bool Copy(DVVectorLike vec_in, DVVectorLike vec_out)
        {
            return Native.copy(vec_in.m_cptr, vec_out.m_cptr);
        }

        public static bool Swap(DVVectorLike vec1, DVVectorLike vec2)
        {
            return Native.swap(vec1.m_cptr, vec2.m_cptr);
        }
    }
}
