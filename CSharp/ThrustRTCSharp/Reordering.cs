using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static long Copy_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor pred, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return (long)Native.copy_if(vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static long Copy_If_Stencil(DVVectorLike vec_in, DVVectorLike vec_stencil, DVVectorLike vec_out, Functor pred, long begin_in = 0, long end_in = -1, long begin_stencil = 0, long begin_out = 0)
        {
            return (long)Native.copy_if_stencil(vec_in.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, pred.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_stencil, (ulong)begin_out);
        }

        public static long Remove(DVVectorLike vec, DeviceViewable value, long begin = 0, long end = -1)
        {
            return (long)Native.remove(vec.m_cptr, value.m_cptr, (ulong)begin, (ulong)end);
        }

        public static long Remove_Copy(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable value, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return (long)Native.remove_copy(vec_in.m_cptr, vec_out.m_cptr, value.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static long Remove_If(DVVectorLike vec, Functor pred, long begin = 0, long end = -1)
        {
            return (long)Native.remove_if(vec.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end);
        }

        public static long Remove_Copy_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor pred, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return (long)Native.remove_copy_if(vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static long Remove_If_Stencil(DVVectorLike vec, DVVectorLike stencil, Functor pred, long begin = 0, long end = -1, long begin_stencil = 0)
        {
            return (long)Native.remove_if_stencil(vec.m_cptr, stencil.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end, (ulong)begin_stencil);
        }

        public static long Remove_Copy_If_Stencil(DVVectorLike vec_in, DVVectorLike stencil, DVVectorLike vec_out, Functor pred, long begin_in = 0, long end_in = -1, long begin_stencil = 0, long begin_out = 0)
        {
            return (long)Native.remove_copy_if_stencil(vec_in.m_cptr, stencil.m_cptr,vec_out.m_cptr, pred.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_stencil, (ulong)begin_out);
        }

        public static long Unique(DVVectorLike vec, long begin = 0, long end = -1)
        {
            return (long)Native.unique(vec.m_cptr, (ulong)begin, (ulong)end);
        }

        public static long Unique(DVVectorLike vec, Functor binary_pred, long begin = 0, long end = -1)
        {
            return (long)Native.unique(vec.m_cptr, binary_pred.m_cptr, (ulong)begin, (ulong)end);
        }

        public static long Unique_Copy(DVVectorLike vec_in, DVVectorLike vec_out, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return (long)Native.unique_copy(vec_in.m_cptr, vec_out.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static long Unique_Copy(DVVectorLike vec_in, DVVectorLike vec_out, Functor binary_pred, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return (long)Native.unique_copy(vec_in.m_cptr, vec_out.m_cptr, binary_pred.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static long Unique_By_Key(DVVectorLike keys, DVVectorLike values, long begin_key = 0, long end_key = -1, long begin_value = 0)
        {
            return (long)Native.unique_by_key(keys.m_cptr, values.m_cptr, (ulong)begin_key, (ulong)end_key, (ulong)begin_value);
        }

        public static long Unique_By_Key(DVVectorLike keys, DVVectorLike values, Functor binary_pred, long begin_key = 0, long end_key = -1, long begin_value = 0)
        {
            return (long)Native.unique_by_key(keys.m_cptr, values.m_cptr, binary_pred.m_cptr, (ulong)begin_key, (ulong)end_key, (ulong)begin_value);
        }

        public static long Unique_By_Key_Copy(DVVectorLike keys_in, DVVectorLike values_in, DVVectorLike keys_out, DVVectorLike values_out, long begin_key_in = 0, long end_key_in = -1, long begin_value_in = 0, long begin_key_out = 0, long begin_value_out = 0)
        {
            return (long)Native.unique_by_key_copy(keys_in.m_cptr, values_in.m_cptr, keys_out.m_cptr, values_out.m_cptr, (ulong)begin_key_in, (ulong)end_key_in, (ulong)begin_value_in, (ulong)begin_key_out, (ulong)begin_value_out);
        }

        public static long Unique_By_Key_Copy(DVVectorLike keys_in, DVVectorLike values_in, DVVectorLike keys_out, DVVectorLike values_out, Functor binary_pred, long begin_key_in = 0, long end_key_in = -1, long begin_value_in = 0, long begin_key_out = 0, long begin_value_out = 0)
        {
            return (long)Native.unique_by_key_copy(keys_in.m_cptr, values_in.m_cptr, keys_out.m_cptr, values_out.m_cptr, binary_pred.m_cptr, (ulong)begin_key_in, (ulong)end_key_in, (ulong)begin_value_in, (ulong)begin_key_out, (ulong)begin_value_out);
        }

        public static long Partition(DVVectorLike vec, Functor pred, long begin = 0, long end = -1)
        {
            return (long)Native.partition(vec.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end);
        }

        public static long Partition_Stencil(DVVectorLike vec, DVVectorLike stencil, Functor pred, long begin = 0, long end = -1, long begin_stencil = 0)
        {
            return (long)Native.partition_stencil(vec.m_cptr, stencil.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end, (ulong)begin_stencil);
        }

        public static long Partition_Copy(DVVectorLike vec_in, DVVectorLike vec_true, DVVectorLike vec_false, Functor pred, long begin_in = 0, long end_in = -1, long begin_true = 0, long begin_false = 0)
        {
            return (long)Native.partition_copy(vec_in.m_cptr, vec_true.m_cptr, vec_false.m_cptr, pred.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_true, (ulong)begin_false);
        }

        public static long Partition_Copy_Stencil(DVVectorLike vec_in, DVVectorLike stencil, DVVectorLike vec_true, DVVectorLike vec_false, Functor pred, long begin_in = 0, long end_in = -1, long begin_stencil = 0, long begin_true = 0, long begin_false = 0)
        {
            return (long)Native.partition_copy_stencil(vec_in.m_cptr, stencil.m_cptr, vec_true.m_cptr, vec_false.m_cptr, pred.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_stencil, (ulong)begin_true, (ulong)begin_false);
        }

    }
}
