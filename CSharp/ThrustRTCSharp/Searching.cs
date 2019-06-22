using System;
using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static object Find(DVVectorLike vec, DeviceViewable value)
        {
            return Native.find(vec.m_cptr, value.m_cptr);
        }

        public static object Find_If(DVVectorLike vec, Functor pred)
        {
            return Native.find_if(vec.m_cptr, pred.m_cptr);
        }

        public static object Find_If_Not(DVVectorLike vec, Functor pred)
        {
            return Native.find_if_not(vec.m_cptr, pred.m_cptr);
        }

        public static object Mismatch(DVVectorLike vec1, DVVectorLike vec2)
        {
            return Native.mismatch(vec1.m_cptr, vec2.m_cptr);
        }

        public static object Mismatch(DVVectorLike vec1, DVVectorLike vec2, Functor pred)
        {
            return Native.mismatch(vec1.m_cptr, vec2.m_cptr, pred.m_cptr);
        }

        public static object Lower_Bound(DVVectorLike vec, DeviceViewable value)
        {
            return Native.lower_bound(vec.m_cptr, value.m_cptr);
        }

        public static object Lower_Bound(DVVectorLike vec, DeviceViewable value, Functor comp)
        {
            return Native.lower_bound(vec.m_cptr, value.m_cptr, comp.m_cptr);
        }

        public static object Upper_Bound(DVVectorLike vec, DeviceViewable value)
        {
            return Native.upper_bound(vec.m_cptr, value.m_cptr);
        }

        public static object Upper_Bound(DVVectorLike vec, DeviceViewable value, Functor comp)
        {
            return Native.upper_bound(vec.m_cptr, value.m_cptr, comp.m_cptr);
        }

        public static object Binary_Search(DVVectorLike vec, DeviceViewable value)
        {
            return Native.binary_search(vec.m_cptr, value.m_cptr);
        }

        public static object Binary_Search(DVVectorLike vec, DeviceViewable value, Functor comp)
        {
            return Native.binary_search(vec.m_cptr, value.m_cptr, comp.m_cptr);
        }

        public static bool Lower_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result)
        {
            return Native.lower_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr);
        }

        public static bool Lower_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, Functor comp)
        {
            return Native.lower_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, comp.m_cptr);
        }

        public static bool Upper_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result)
        {
            return Native.upper_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr);
        }

        public static bool Upper_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, Functor comp)
        {
            return Native.upper_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, comp.m_cptr);
        }

        public static bool Binary_Search_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result)
        {
            return Native.binary_search_v(vec.m_cptr, values.m_cptr, result.m_cptr);
        }

        public static bool Binary_Search_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, Functor comp)
        {
            return Native.binary_search_v(vec.m_cptr, values.m_cptr, result.m_cptr, comp.m_cptr);
        }

        public static object Partition_Point(DVVectorLike vec, Functor pred)
        {
            return Native.partition_point(vec.m_cptr, pred.m_cptr);
        }

        public static object Is_Sorted_Until(DVVectorLike vec)
        {
            return Native.is_sorted_until(vec.m_cptr);
        }

        public static object Is_Sorted_Until(DVVectorLike vec, Functor comp)
        {
            return Native.is_sorted_until(vec.m_cptr, comp.m_cptr);
        }
    }
}
