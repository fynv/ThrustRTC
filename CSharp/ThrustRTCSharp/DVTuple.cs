using System;
using ThrustRTCLR;
using System.Runtime.InteropServices;

namespace ThrustRTCSharp
{
    public class DVTuple : DeviceViewable
    {
        private static IntPtr create(DeviceViewable[] objs, string[] name_objs)
        {
            CapturedDeviceViewable_clr[] arg_map = new CapturedDeviceViewable_clr[objs.Length];
            for (int i = 0; i < objs.Length; i++)
            {
                arg_map[i].obj = objs[i].m_cptr;
                arg_map[i].obj_name = Marshal.StringToHGlobalAnsi(name_objs[i]);
            }

            return Native.dvtuple_create(arg_map);
        }

        public DVTuple(DeviceViewable[] objs, string[] name_objs) : base(create(objs, name_objs))
        {
            m_objs = objs;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposed) return;
            m_objs = null;
            base.Dispose(disposing);
        }

        private DeviceViewable[] m_objs;
    }
}
