using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static bool Gather(DVVectorLike vec_map, DVVectorLike vec_in, DVVectorLike vec_out, long begin_map = 0, long end_map = -1, long begin_in = 0, long begin_out = 0)
        {
            return Native.gather(vec_map.m_cptr, vec_in.m_cptr, vec_out.m_cptr, (ulong)begin_map, (ulong)end_map, (ulong)begin_in, (ulong)begin_out);
        }

        public static bool Gather_If(DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_in, DVVectorLike vec_out, long begin_map = 0, long end_map = -1, long begin_stencil = 0, long begin_in = 0, long begin_out = 0)
        {
            return Native.gather_if(vec_map.m_cptr, vec_stencil.m_cptr, vec_in.m_cptr, vec_out.m_cptr, (ulong)begin_map, (ulong)end_map, (ulong)begin_stencil, (ulong)begin_in, (ulong)begin_out);
        }

        public static bool Gather_If(DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_in, DVVectorLike vec_out, Functor pred, long begin_map = 0, long end_map = -1, long begin_stencil = 0, long begin_in = 0, long begin_out = 0)
        {
            return Native.gather_if(vec_map.m_cptr, vec_stencil.m_cptr, vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr, (ulong)begin_map, (ulong)end_map, (ulong)begin_stencil, (ulong)begin_in, (ulong)begin_out);
        }

        public static bool Scatter(DVVectorLike vec_in, DVVectorLike vec_map, DVVectorLike vec_out, long begin_in = 0, long end_in = -1, long begin_map = 0, long begin_out = 0)
        {
            return Native.scatter(vec_in.m_cptr, vec_map.m_cptr, vec_out.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_map, (ulong)begin_out);
        }

        public static bool Scatter_If(DVVectorLike vec_in, DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_out, long begin_in = 0, long end_in = -1, long begin_map = 0, long begin_stencil = 0, long begin_out = 0)
        {
            return Native.scatter_if(vec_in.m_cptr, vec_map.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_map, (ulong)begin_stencil, (ulong)begin_out);
        }

        public static bool Scatter_If(DVVectorLike vec_in, DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_out, Functor pred, long begin_in = 0, long end_in = -1, long begin_map = 0, long begin_stencil = 0, long begin_out = 0)
        {
            return Native.scatter_if(vec_in.m_cptr, vec_map.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, pred.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_map, (ulong)begin_stencil, (ulong)begin_out);
        }

        public static bool Copy(DVVectorLike vec_in, DVVectorLike vec_out, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.copy(vec_in.m_cptr, vec_out.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Swap(DVVectorLike vec1, DVVectorLike vec2, long begin1 = 0, long end1 = -1, long begin2 = 0)
        {
            return Native.swap(vec1.m_cptr, vec2.m_cptr, (ulong)begin1, (ulong)end1, (ulong)begin2);
        }
    }
}
