using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCLR;
using System.Runtime.InteropServices;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static long Count(DVVectorLike vec, DeviceViewable value, long begin = 0, long end = -1)
        {
            return (long)Native.count(vec.m_cptr, value.m_cptr, (ulong)begin, (ulong)end);
        }

        public static long Count_If(DVVectorLike vec, Functor pred, long begin = 0, long end = -1)
        {
            return (long)Native.count_if(vec.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Reduce(DVVectorLike vec, long begin = 0, long end = -1)
        {
            return Native.reduce(vec.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Reduce(DVVectorLike vec, DeviceViewable init, long begin = 0, long end = -1)
        {
            return Native.reduce(vec.m_cptr, init.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Reduce(DVVectorLike vec, DeviceViewable init, Functor binary_op, long begin = 0, long end = -1)
        {
            return Native.reduce(vec.m_cptr, init.m_cptr, binary_op.m_cptr, (ulong)begin, (ulong)end);
        }

        public static long Reduce_By_Key(DVVectorLike key_in, DVVectorLike value_in, DVVectorLike key_out, DVVectorLike value_out, long begin_key_in = 0, long end_key_in = -1, long begin_value_in = 0, long begin_key_out = 0, long begin_value_out = 0)
        {
            return (long)Native.reduce_by_key(key_in.m_cptr, value_in.m_cptr, key_out.m_cptr, value_out.m_cptr, (ulong)begin_key_in, (ulong)end_key_in, (ulong)begin_value_in, (ulong)begin_key_out, (ulong)begin_value_out);
        }

        public static long Reduce_By_Key(DVVectorLike key_in, DVVectorLike value_in, DVVectorLike key_out, DVVectorLike value_out, Functor binary_pred, long begin_key_in = 0, long end_key_in = -1, long begin_value_in = 0, long begin_key_out = 0, long begin_value_out = 0)
        {
            return (long)Native.reduce_by_key(key_in.m_cptr, value_in.m_cptr, key_out.m_cptr, value_out.m_cptr, binary_pred.m_cptr, (ulong)begin_key_in, (ulong)end_key_in, (ulong)begin_value_in, (ulong)begin_key_out, (ulong)begin_value_out);
        }

        public static long Reduce_By_Key(DVVectorLike key_in, DVVectorLike value_in, DVVectorLike key_out, DVVectorLike value_out, Functor binary_pred, Functor binary_op, long begin_key_in = 0, long end_key_in = -1, long begin_value_in = 0, long begin_key_out = 0, long begin_value_out = 0)
        {
            return (long)Native.reduce_by_key(key_in.m_cptr, value_in.m_cptr, key_out.m_cptr, value_out.m_cptr, binary_pred.m_cptr, binary_op.m_cptr, (ulong)begin_key_in, (ulong)end_key_in, (ulong)begin_value_in, (ulong)begin_key_out, (ulong)begin_value_out);
        }

        public static object Equal(DVVectorLike vec1, DVVectorLike vec2, long begin1 = 0, long end1 = -1, long begin2 = 0)
        {
            return Native.equal(vec1.m_cptr, vec2.m_cptr, (ulong)begin1, (ulong)end1, (ulong)begin2);
        }

        public static object Equal(DVVectorLike vec1, DVVectorLike vec2, Functor binary_pred, long begin1 = 0, long end1 = -1, long begin2 = 0)
        {
            return Native.equal(vec1.m_cptr, vec2.m_cptr, binary_pred.m_cptr, (ulong)begin1, (ulong)end1, (ulong)begin2);
        }

        public static long Min_Element(DVVectorLike vec, long begin = 0, long end = -1)
        {
            return (long)Native.min_element(vec.m_cptr, (ulong)begin, (ulong)end);
        }

        public static long Min_Element(DVVectorLike vec, Functor comp, long begin = 0, long end = -1)
        {
            return (long)Native.min_element(vec.m_cptr, comp.m_cptr, (ulong)begin, (ulong)end);
        }

        public static long Max_Element(DVVectorLike vec, long begin = 0, long end = -1)
        {
            return (long)Native.max_element(vec.m_cptr, (ulong)begin, (ulong)end);
        }

        public static long Max_Element(DVVectorLike vec, Functor comp, long begin = 0, long end = -1)
        {
            return (long)Native.max_element(vec.m_cptr, comp.m_cptr, (ulong)begin, (ulong)end);
        }

        public static Tuple<long, long> MinMax_Element(DVVectorLike vec, long begin = 0, long end = -1)
        {
            return Native.minmax_element(vec.m_cptr, (ulong)begin, (ulong)end);
        }

        public static Tuple<long, long> MinMax_Element(DVVectorLike vec, Functor comp, long begin = 0, long end = -1)
        {
            return Native.minmax_element(vec.m_cptr, comp.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Inner_Product(DVVectorLike vec1, DVVectorLike vec2, DeviceViewable init, long begin1 = 0, long end1 = -1, long begin2 = 0)
        {
            return Native.inner_product(vec1.m_cptr, vec2.m_cptr, init.m_cptr, (ulong)begin1, (ulong)end1, (ulong)begin2);
        }

        public static object Inner_Product(DVVectorLike vec1, DVVectorLike vec2, DeviceViewable init, Functor binary_op1, Functor binary_op2, long begin1 = 0, long end1 = -1, long begin2 = 0)
        {
            return Native.inner_product(vec1.m_cptr, vec2.m_cptr, init.m_cptr, binary_op1.m_cptr, binary_op2.m_cptr, (ulong)begin1, (ulong)end1, (ulong)begin2);
        }

        public static object Transform_Reduce(DVVectorLike vec, Functor unary_op, DeviceViewable init, Functor binary_op, long begin = 0, long end = -1)
        {
            return Native.transform_reduce(vec.m_cptr, unary_op.m_cptr, init.m_cptr, binary_op.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object All_Of(DVVectorLike vec, Functor pred, long begin = 0, long end = -1)
        {
            return Native.all_of(vec.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Any_Of(DVVectorLike vec, Functor pred, long begin = 0, long end = -1)
        {
            return Native.any_of(vec.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object None_Of(DVVectorLike vec, Functor pred, long begin = 0, long end = -1)
        {
            return Native.none_of(vec.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Is_Partitioned(DVVectorLike vec, Functor pred, long begin = 0, long end = -1)
        {
            return Native.is_partitioned(vec.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Is_Sorted(DVVectorLike vec, long begin = 0, long end = -1)
        {
            return Native.is_sorted(vec.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Is_Sorted(DVVectorLike vec, Functor comp, long begin = 0, long end = -1)
        {
            return Native.is_sorted(vec.m_cptr, comp.m_cptr, (ulong)begin, (ulong)end);
        }

    }
}
