using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static bool Fill(DVVectorLike vec, DeviceViewable value)
        {
            return Native.fiil(vec.m_cptr, value.m_cptr);
        }

        public static bool Replace(DVVectorLike vec, DeviceViewable old_value, DeviceViewable new_value)
        {
            return Native.replace(vec.m_cptr, old_value.m_cptr, new_value.m_cptr);
        }

        public static bool Replace_If(DVVectorLike vec, Functor pred, DeviceViewable new_value)
        {
            return Native.replace_if(vec.m_cptr, pred.m_cptr, new_value.m_cptr);
        }

        public static bool Replace_Copy(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable old_value, DeviceViewable new_value)
        {
            return Native.replace_copy(vec_in.m_cptr, vec_out.m_cptr, old_value.m_cptr, new_value.m_cptr);
        }

        public static bool Replace_Copy_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor pred, DeviceViewable new_value)
        {
            return Native.replace_copy_if(vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr, new_value.m_cptr);
        }

        public static bool For_Each(DVVectorLike vec, Functor f)
        {
            return Native.for_each(vec.m_cptr, f.m_cptr);
        }

        public static bool Adjacent_Difference(DVVectorLike vec_in, DVVectorLike vec_out)
        {
            return Native.adjacent_difference(vec_in.m_cptr, vec_out.m_cptr);
        }

        public static bool Adjacent_Difference(DVVectorLike vec_in, DVVectorLike vec_out, Functor f)
        {
            return Native.adjacent_difference(vec_in.m_cptr, vec_out.m_cptr, f.m_cptr);
        }

        public static bool Sequence(DVVectorLike vec)
        {
            return Native.sequence(vec.m_cptr);
        }

        public static bool Sequence(DVVectorLike vec, DeviceViewable value_init)
        {
            return Native.sequence(vec.m_cptr, value_init.m_cptr);
        }

        public static bool Sequence(DVVectorLike vec, DeviceViewable value_init, DeviceViewable value_step)
        {
            return Native.sequence(vec.m_cptr, value_init.m_cptr, value_step.m_cptr);
        }

        public static bool Tabulate(DVVectorLike vec, Functor op)
        {
            return Native.tabulate(vec.m_cptr, op.m_cptr);
        }

        public static bool Transform(DVVectorLike vec_in, DVVectorLike vec_out, Functor op)
        {
            return Native.transform(vec_in.m_cptr, vec_out.m_cptr, op.m_cptr);
        }

        public static bool Transform_Binary(DVVectorLike vec_in1, DVVectorLike vec_in2, DVVectorLike vec_out, Functor op)
        {
            return Native.transform_binary(vec_in1.m_cptr, vec_in2.m_cptr, vec_out.m_cptr, op.m_cptr);
        }

        public static bool Transform_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor op, Functor pred)
        {
            return Native.transform_if(vec_in.m_cptr, vec_out.m_cptr, op.m_cptr, pred.m_cptr);
        }

        public static bool Transform_If_Stencil(DVVectorLike vec_in, DVVectorLike vec_stencil,  DVVectorLike vec_out, Functor op, Functor pred)
        {
            return Native.transform_if_stencil(vec_in.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, op.m_cptr, pred.m_cptr);
        }

        public static bool Transform_Binary_If_Stencil(DVVectorLike vec_in1, DVVectorLike vec_in2, DVVectorLike vec_stencil, DVVectorLike vec_out, Functor op, Functor pred)
        {
            return Native.transform_binary_if_stencil(vec_in1.m_cptr, vec_in2.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, op.m_cptr, pred.m_cptr);
        }
    }

}
