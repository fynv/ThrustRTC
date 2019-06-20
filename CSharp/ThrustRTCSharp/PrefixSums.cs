using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        public static bool Inclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out)
        {
            return Native.inclusive_scan(vec_in.m_cptr, vec_out.m_cptr);
        }

        public static bool Inclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, Functor binary_op)
        {
            return Native.inclusive_scan(vec_in.m_cptr, vec_out.m_cptr, binary_op.m_cptr);
        }

        public static bool Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out)
        {
            return Native.exclusive_scan(vec_in.m_cptr, vec_out.m_cptr);
        }

        public static bool Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable init)
        {
            return Native.exclusive_scan(vec_in.m_cptr, vec_out.m_cptr, init.m_cptr);
        }

        public static bool Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable init, Functor binary_op)
        {
            return Native.exclusive_scan(vec_in.m_cptr, vec_out.m_cptr, init.m_cptr, binary_op.m_cptr);
        }

        public static bool Inclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out)
        {
            return Native.inclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr);
        }

        public static bool Inclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, Functor binary_pred)
        {
            return Native.inclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, binary_pred.m_cptr);
        }

        public static bool Inclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, Functor binary_pred, Functor binary_op)
        {
            return Native.inclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, binary_pred.m_cptr, binary_op.m_cptr);
        }

        public static bool Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out)
        {
            return Native.exclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr);
        }

        public static bool Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, DeviceViewable init)
        {
            return Native.exclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, init.m_cptr);
        }

        public static bool Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, DeviceViewable init, Functor binary_pred)
        {
            return Native.exclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, init.m_cptr, binary_pred.m_cptr);
        }

        public static bool Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, DeviceViewable init, Functor binary_pred, Functor binary_op)
        {
            return Native.exclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, init.m_cptr, binary_pred.m_cptr, binary_op.m_cptr);
        }

        public static bool Transform_Inclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, Functor unary_op, Functor binary_op)
        {
            return Native.transform_inclusive_scan(vec_in.m_cptr, vec_out.m_cptr, unary_op.m_cptr, binary_op.m_cptr);
        }

        public static bool Transform_Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, Functor unary_op, DeviceViewable init, Functor binary_op)
        {
            return Native.transform_exclusive_scan(vec_in.m_cptr, vec_out.m_cptr, unary_op.m_cptr, init.m_cptr, binary_op.m_cptr);
        }
    }
}
