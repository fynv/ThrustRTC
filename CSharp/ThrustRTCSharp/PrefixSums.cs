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
        public static bool Inclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.inclusive_scan(vec_in.m_cptr, vec_out.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Inclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, Functor binary_op, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.inclusive_scan(vec_in.m_cptr, vec_out.m_cptr, binary_op.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.exclusive_scan(vec_in.m_cptr, vec_out.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable init, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.exclusive_scan(vec_in.m_cptr, vec_out.m_cptr, init.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable init, Functor binary_op, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.exclusive_scan(vec_in.m_cptr, vec_out.m_cptr, init.m_cptr, binary_op.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Inclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, long begin_key = 0, long end_key = -1, long begin_value =0, long begin_out=0)
        {
            return Native.inclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, (ulong)begin_key, (ulong)end_key, (ulong)begin_value, (ulong)begin_out);
        }

        public static bool Inclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, Functor binary_pred, long begin_key = 0, long end_key = -1, long begin_value = 0, long begin_out = 0)
        {
            return Native.inclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, binary_pred.m_cptr, (ulong)begin_key, (ulong)end_key, (ulong)begin_value, (ulong)begin_out);
        }

        public static bool Inclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, Functor binary_pred, Functor binary_op, long begin_key = 0, long end_key = -1, long begin_value = 0, long begin_out = 0)
        {
            return Native.inclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, binary_pred.m_cptr, binary_op.m_cptr, (ulong)begin_key, (ulong)end_key, (ulong)begin_value, (ulong)begin_out);
        }

        public static bool Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, long begin_key = 0, long end_key = -1, long begin_value = 0, long begin_out = 0)
        {
            return Native.exclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, (ulong)begin_key, (ulong)end_key, (ulong)begin_value, (ulong)begin_out);
        }

        public static bool Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, DeviceViewable init, long begin_key = 0, long end_key = -1, long begin_value = 0, long begin_out = 0)
        {
            return Native.exclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, init.m_cptr, (ulong)begin_key, (ulong)end_key, (ulong)begin_value, (ulong)begin_out);
        }

        public static bool Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, DeviceViewable init, Functor binary_pred, long begin_key = 0, long end_key = -1, long begin_value = 0, long begin_out = 0)
        {
            return Native.exclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, init.m_cptr, binary_pred.m_cptr, (ulong)begin_key, (ulong)end_key, (ulong)begin_value, (ulong)begin_out);
        }

        public static bool Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, DeviceViewable init, Functor binary_pred, Functor binary_op, long begin_key = 0, long end_key = -1, long begin_value = 0, long begin_out = 0)
        {
            return Native.exclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, init.m_cptr, binary_pred.m_cptr, binary_op.m_cptr, (ulong)begin_key, (ulong)end_key, (ulong)begin_value, (ulong)begin_out);
        }

        public static bool Transform_Inclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, Functor unary_op, Functor binary_op, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.transform_inclusive_scan(vec_in.m_cptr, vec_out.m_cptr, unary_op.m_cptr, binary_op.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }

        public static bool Transform_Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, Functor unary_op, DeviceViewable init, Functor binary_op, long begin_in = 0, long end_in = -1, long begin_out = 0)
        {
            return Native.transform_exclusive_scan(vec_in.m_cptr, vec_out.m_cptr, unary_op.m_cptr, init.m_cptr, binary_op.m_cptr, (ulong)begin_in, (ulong)end_in, (ulong)begin_out);
        }
    }
}
