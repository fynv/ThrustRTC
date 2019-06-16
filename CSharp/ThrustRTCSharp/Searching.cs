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
        public static object Find(DVVectorLike vec, DeviceViewable value, long begin = 0, long end = -1)
        {
            return Native.find(vec.m_cptr, value.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Find_If(DVVectorLike vec, Functor pred, long begin = 0, long end = -1)
        {
            return Native.find_if(vec.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Find_If_Not(DVVectorLike vec, Functor pred, long begin = 0, long end = -1)
        {
            return Native.find_if_not(vec.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end);
        }

        public static Tuple<long, long> Mismatch(DVVectorLike vec1, DVVectorLike vec2, long begin1 = 0, long end1 = -1, long begin2 = 0)
        {
            return Native.mismatch(vec1.m_cptr, vec2.m_cptr, (ulong)begin1, (ulong)end1, (ulong)begin2);
        }

        public static Tuple<long, long> Mismatch(DVVectorLike vec1, DVVectorLike vec2, Functor pred, long begin1 = 0, long end1 = -1, long begin2 = 0)
        {
            return Native.mismatch(vec1.m_cptr, vec2.m_cptr, pred.m_cptr, (ulong)begin1, (ulong)end1, (ulong)begin2);
        }

        public static object Lower_Bound(DVVectorLike vec, DeviceViewable value, long begin = 0, long end = -1)
        {
            return Native.lower_bound(vec.m_cptr, value.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Lower_Bound(DVVectorLike vec, DeviceViewable value, Functor comp, long begin = 0, long end = -1)
        {
            return Native.lower_bound(vec.m_cptr, value.m_cptr, comp.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Upper_Bound(DVVectorLike vec, DeviceViewable value, long begin = 0, long end = -1)
        {
            return Native.upper_bound(vec.m_cptr, value.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Upper_Bound(DVVectorLike vec, DeviceViewable value, Functor comp, long begin = 0, long end = -1)
        {
            return Native.upper_bound(vec.m_cptr, value.m_cptr, comp.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Binary_Search(DVVectorLike vec, DeviceViewable value, long begin = 0, long end = -1)
        {
            return Native.binary_search(vec.m_cptr, value.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Binary_Search(DVVectorLike vec, DeviceViewable value, Functor comp, long begin = 0, long end = -1)
        {
            return Native.binary_search(vec.m_cptr, value.m_cptr, comp.m_cptr, (ulong)begin, (ulong)end);
        }

        public static object Lower_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, long begin = 0, long end = -1, long begin_values = 0, long end_values = -1, long begin_result = 0)
        {
            return Native.lower_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, (ulong)begin, (ulong)end, (ulong)begin_values, (ulong)(end_values), (ulong)begin_result);
        }

        public static object Lower_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, Functor comp, long begin = 0, long end = -1, long begin_values = 0, long end_values = -1, long begin_result = 0)
        {
            return Native.lower_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, comp.m_cptr, (ulong)begin, (ulong)end, (ulong)begin_values, (ulong)(end_values), (ulong)begin_result);
        }

        public static object Upper_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, long begin = 0, long end = -1, long begin_values = 0, long end_values = -1, long begin_result = 0)
        {
            return Native.upper_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, (ulong)begin, (ulong)end, (ulong)begin_values, (ulong)(end_values), (ulong)begin_result);
        }

        public static object Upper_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, Functor comp, long begin = 0, long end = -1, long begin_values = 0, long end_values = -1, long begin_result = 0)
        {
            return Native.upper_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, comp.m_cptr, (ulong)begin, (ulong)end, (ulong)begin_values, (ulong)(end_values), (ulong)begin_result);
        }

        public static object Binary_Search_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, long begin = 0, long end = -1, long begin_values = 0, long end_values = -1, long begin_result = 0)
        {
            return Native.binary_search_v(vec.m_cptr, values.m_cptr, result.m_cptr, (ulong)begin, (ulong)end, (ulong)begin_values, (ulong)(end_values), (ulong)begin_result);
        }

        public static object Binary_Search_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, Functor comp, long begin = 0, long end = -1, long begin_values = 0, long end_values = -1, long begin_result = 0)
        {
            return Native.binary_search_v(vec.m_cptr, values.m_cptr, result.m_cptr, comp.m_cptr, (ulong)begin, (ulong)end, (ulong)begin_values, (ulong)(end_values), (ulong)begin_result);
        }

        public static object Partition_Point(DVVectorLike vec, Functor pred, long begin = 0, long end = -1)
        {
            return Native.partition_point(vec.m_cptr, pred.m_cptr, (ulong)begin, (ulong)end);
        }
    }
}
