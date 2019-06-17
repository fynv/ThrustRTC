using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static bool Fill(DVVectorLike vec, DeviceViewable value, long begin = 0, long end = -1)
        {
            return Native.fiil(vec.m_cptr, value.m_cptr, (ulong)begin, (ulong)end);
        }

        public static bool Replace(DVVectorLike vec, DeviceViewable old_value, DeviceViewable new_value, long begin = 0, long end = -1)
        {
            return Native.replace(vec.m_cptr, old_value.m_cptr, new_value.m_cptr, (ulong)begin, (ulong)end);
        }

        public static bool Replace_If(DVVectorLike vec, Functor pred, DeviceViewable new_value, long begin = 0, long end = -1)
        {
            return Native.replace_if(vec.m_cptr, pred.m_cptr, new_value.m_cptr, (ulong)begin, (ulong)end);
        }

        public static bool Replace_Copy(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable old_value, DeviceViewable new_value, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.replace_copy(vec_in.m_cptr, vec_out.m_cptr, old_value.m_cptr, new_value.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Replace_Copy_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor pred, DeviceViewable new_value, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.replace_copy_if(vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr, new_value.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool For_Each(DVVectorLike vec, Functor f, long begin = 0, long end = -1)
        {
            return Native.for_each(vec.m_cptr, f.m_cptr, (ulong)begin, (ulong)end);
        }

        public static bool Adjacent_Difference(DVVectorLike vec_in, DVVectorLike vec_out, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.adjacent_difference(vec_in.m_cptr, vec_out.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Adjacent_Difference(DVVectorLike vec_in, DVVectorLike vec_out, Functor f, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.adjacent_difference(vec_in.m_cptr, vec_out.m_cptr, f.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Sequence(DVVectorLike vec, long begin = 0, long end = -1)
        {
            return Native.sequence(vec.m_cptr, (ulong)begin, (ulong)end);
        }

        public static bool Sequence(DVVectorLike vec, DeviceViewable value_init, long begin = 0, long end = -1)
        {
            return Native.sequence(vec.m_cptr, value_init.m_cptr, (ulong)begin, (ulong)end);
        }

        public static bool Sequence(DVVectorLike vec, DeviceViewable value_init, DeviceViewable value_step, long begin = 0, long end = -1)
        {
            return Native.sequence(vec.m_cptr, value_init.m_cptr, value_step.m_cptr, (ulong)begin, (ulong)end);
        }

        public static bool Tabulate(DVVectorLike vec, Functor op, long begin = 0, long end = -1)
        {
            return Native.tabulate(vec.m_cptr, op.m_cptr, (ulong)begin, (ulong)end);
        }

        public static bool Transform(DVVectorLike vec_in, DVVectorLike vec_out, Functor op, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.transform(vec_in.m_cptr, vec_out.m_cptr, op.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Transform_Binary(DVVectorLike vec_in1, DVVectorLike vec_in2, DVVectorLike vec_out, Functor op, long begin_in1 = 0, long end_in1 = -1, long begin_in2 = 0, long begin_out = 0)
        {
            return Native.transform_binary(vec_in1.m_cptr, vec_in2.m_cptr, vec_out.m_cptr, op.m_cptr, (ulong)begin_in1, (ulong)end_in1, (ulong)begin_in2, (ulong)begin_out);
        }

        public static bool Transform_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor op, Functor pred, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.transform_if(vec_in.m_cptr, vec_out.m_cptr, op.m_cptr, pred.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Transform_If_Stencil(DVVectorLike vec_in, DVVectorLike vec_stencil,  DVVectorLike vec_out, Functor op, Functor pred, long begin_in = 0, long end_in = -1, long begin_stencil = 0, long begin_out = 0)
        {
            return Native.transform_if_stencil(vec_in.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, op.m_cptr, pred.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_stencil, (ulong)begin_out);
        }

        public static bool Transform_Binary_If_Stencil(DVVectorLike vec_in1, DVVectorLike vec_in2, DVVectorLike vec_stencil, DVVectorLike vec_out, Functor op, Functor pred, long begin_in1 = 0, long end_in1 = -1, long begin_in2 =0 ,long begin_stencil = 0, long begin_out = 0)
        {
            return Native.transform_binary_if_stencil(vec_in1.m_cptr, vec_in2.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, op.m_cptr, pred.m_cptr, (ulong)begin_in1, (ulong)end_in1, (ulong)begin_in2, (ulong)begin_stencil, (ulong)begin_out);
        }
    }

}
