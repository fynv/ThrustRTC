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
    }

}
