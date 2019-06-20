using System;
using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static long Count(DVVectorLike vec, DeviceViewable value)
        {
            return (long)Native.count(vec.m_cptr, value.m_cptr);
        }

        public static long Count_If(DVVectorLike vec, Functor pred)
        {
            return (long)Native.count_if(vec.m_cptr, pred.m_cptr);
        }

        public static object Reduce(DVVectorLike vec)
        {
            return Native.reduce(vec.m_cptr);
        }

        public static object Reduce(DVVectorLike vec, DeviceViewable init)
        {
            return Native.reduce(vec.m_cptr, init.m_cptr);
        }

        public static object Reduce(DVVectorLike vec, DeviceViewable init, Functor binary_op)
        {
            return Native.reduce(vec.m_cptr, init.m_cptr, binary_op.m_cptr);
        }

        public static long Reduce_By_Key(DVVectorLike key_in, DVVectorLike value_in, DVVectorLike key_out, DVVectorLike value_out)
        {
            return (long)Native.reduce_by_key(key_in.m_cptr, value_in.m_cptr, key_out.m_cptr, value_out.m_cptr);
        }

        public static long Reduce_By_Key(DVVectorLike key_in, DVVectorLike value_in, DVVectorLike key_out, DVVectorLike value_out, Functor binary_pred)
        {
            return (long)Native.reduce_by_key(key_in.m_cptr, value_in.m_cptr, key_out.m_cptr, value_out.m_cptr, binary_pred.m_cptr);
        }

        public static long Reduce_By_Key(DVVectorLike key_in, DVVectorLike value_in, DVVectorLike key_out, DVVectorLike value_out, Functor binary_pred, Functor binary_op)
        {
            return (long)Native.reduce_by_key(key_in.m_cptr, value_in.m_cptr, key_out.m_cptr, value_out.m_cptr, binary_pred.m_cptr, binary_op.m_cptr);
        }

        public static object Equal(DVVectorLike vec1, DVVectorLike vec2)
        {
            return Native.equal(vec1.m_cptr, vec2.m_cptr);
        }

        public static object Equal(DVVectorLike vec1, DVVectorLike vec2, Functor binary_pred)
        {
            return Native.equal(vec1.m_cptr, vec2.m_cptr, binary_pred.m_cptr);
        }

        public static long Min_Element(DVVectorLike vec)
        {
            return (long)Native.min_element(vec.m_cptr);
        }

        public static long Min_Element(DVVectorLike vec, Functor comp)
        {
            return (long)Native.min_element(vec.m_cptr, comp.m_cptr);
        }

        public static long Max_Element(DVVectorLike vec)
        {
            return (long)Native.max_element(vec.m_cptr);
        }

        public static long Max_Element(DVVectorLike vec, Functor comp)
        {
            return (long)Native.max_element(vec.m_cptr, comp.m_cptr);
        }

        public static Tuple<long, long> MinMax_Element(DVVectorLike vec)
        {
            return Native.minmax_element(vec.m_cptr);
        }

        public static Tuple<long, long> MinMax_Element(DVVectorLike vec, Functor comp)
        {
            return Native.minmax_element(vec.m_cptr, comp.m_cptr);
        }

        public static object Inner_Product(DVVectorLike vec1, DVVectorLike vec2, DeviceViewable init)
        {
            return Native.inner_product(vec1.m_cptr, vec2.m_cptr, init.m_cptr);
        }

        public static object Inner_Product(DVVectorLike vec1, DVVectorLike vec2, DeviceViewable init, Functor binary_op1, Functor binary_op2)
        {
            return Native.inner_product(vec1.m_cptr, vec2.m_cptr, init.m_cptr, binary_op1.m_cptr, binary_op2.m_cptr);
        }

        public static object Transform_Reduce(DVVectorLike vec, Functor unary_op, DeviceViewable init, Functor binary_op)
        {
            return Native.transform_reduce(vec.m_cptr, unary_op.m_cptr, init.m_cptr, binary_op.m_cptr);
        }

        public static object All_Of(DVVectorLike vec, Functor pred)
        {
            return Native.all_of(vec.m_cptr, pred.m_cptr);
        }

        public static object Any_Of(DVVectorLike vec, Functor pred)
        {
            return Native.any_of(vec.m_cptr, pred.m_cptr);
        }

        public static object None_Of(DVVectorLike vec, Functor pred)
        {
            return Native.none_of(vec.m_cptr, pred.m_cptr);
        }

        public static object Is_Partitioned(DVVectorLike vec, Functor pred)
        {
            return Native.is_partitioned(vec.m_cptr, pred.m_cptr);
        }

        public static object Is_Sorted(DVVectorLike vec)
        {
            return Native.is_sorted(vec.m_cptr);
        }

        public static object Is_Sorted(DVVectorLike vec, Functor comp)
        {
            return Native.is_sorted(vec.m_cptr, comp.m_cptr);
        }

    }
}
