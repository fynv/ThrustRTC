using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static long Copy_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor pred)
        {
            return (long)Native.copy_if(vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr);
        }

        public static long Copy_If_Stencil(DVVectorLike vec_in, DVVectorLike vec_stencil, DVVectorLike vec_out, Functor pred)
        {
            return (long)Native.copy_if_stencil(vec_in.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, pred.m_cptr);
        }

        public static long Remove(DVVectorLike vec, DeviceViewable value)
        {
            return (long)Native.remove(vec.m_cptr, value.m_cptr);
        }

        public static long Remove_Copy(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable value)
        {
            return (long)Native.remove_copy(vec_in.m_cptr, vec_out.m_cptr, value.m_cptr);
        }

        public static long Remove_If(DVVectorLike vec, Functor pred)
        {
            return (long)Native.remove_if(vec.m_cptr, pred.m_cptr);
        }

        public static long Remove_Copy_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor pred)
        {
            return (long)Native.remove_copy_if(vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr);
        }

        public static long Remove_If_Stencil(DVVectorLike vec, DVVectorLike stencil, Functor pred)
        {
            return (long)Native.remove_if_stencil(vec.m_cptr, stencil.m_cptr, pred.m_cptr);
        }

        public static long Remove_Copy_If_Stencil(DVVectorLike vec_in, DVVectorLike stencil, DVVectorLike vec_out, Functor pred)
        {
            return (long)Native.remove_copy_if_stencil(vec_in.m_cptr, stencil.m_cptr,vec_out.m_cptr, pred.m_cptr);
        }

        public static long Unique(DVVectorLike vec)
        {
            return (long)Native.unique(vec.m_cptr);
        }

        public static long Unique(DVVectorLike vec, Functor binary_pred)
        {
            return (long)Native.unique(vec.m_cptr, binary_pred.m_cptr);
        }

        public static long Unique_Copy(DVVectorLike vec_in, DVVectorLike vec_out)
        {
            return (long)Native.unique_copy(vec_in.m_cptr, vec_out.m_cptr);
        }

        public static long Unique_Copy(DVVectorLike vec_in, DVVectorLike vec_out, Functor binary_pred)
        {
            return (long)Native.unique_copy(vec_in.m_cptr, vec_out.m_cptr, binary_pred.m_cptr);
        }

        public static long Unique_By_Key(DVVectorLike keys, DVVectorLike values)
        {
            return (long)Native.unique_by_key(keys.m_cptr, values.m_cptr);
        }

        public static long Unique_By_Key(DVVectorLike keys, DVVectorLike values, Functor binary_pred)
        {
            return (long)Native.unique_by_key(keys.m_cptr, values.m_cptr, binary_pred.m_cptr);
        }

        public static long Unique_By_Key_Copy(DVVectorLike keys_in, DVVectorLike values_in, DVVectorLike keys_out, DVVectorLike values_out)
        {
            return (long)Native.unique_by_key_copy(keys_in.m_cptr, values_in.m_cptr, keys_out.m_cptr, values_out.m_cptr);
        }

        public static long Unique_By_Key_Copy(DVVectorLike keys_in, DVVectorLike values_in, DVVectorLike keys_out, DVVectorLike values_out, Functor binary_pred)
        {
            return (long)Native.unique_by_key_copy(keys_in.m_cptr, values_in.m_cptr, keys_out.m_cptr, values_out.m_cptr, binary_pred.m_cptr);
        }

        public static long Partition(DVVectorLike vec, Functor pred)
        {
            return (long)Native.partition(vec.m_cptr, pred.m_cptr);
        }

        public static long Partition_Stencil(DVVectorLike vec, DVVectorLike stencil, Functor pred)
        {
            return (long)Native.partition_stencil(vec.m_cptr, stencil.m_cptr, pred.m_cptr);
        }

        public static long Partition_Copy(DVVectorLike vec_in, DVVectorLike vec_true, DVVectorLike vec_false, Functor pred)
        {
            return (long)Native.partition_copy(vec_in.m_cptr, vec_true.m_cptr, vec_false.m_cptr, pred.m_cptr);
        }

        public static long Partition_Copy_Stencil(DVVectorLike vec_in, DVVectorLike stencil, DVVectorLike vec_true, DVVectorLike vec_false, Functor pred)
        {
            return (long)Native.partition_copy_stencil(vec_in.m_cptr, stencil.m_cptr, vec_true.m_cptr, vec_false.m_cptr, pred.m_cptr);
        }

    }
}
