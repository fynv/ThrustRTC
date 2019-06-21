using System;
using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public class DeviceViewable : IDisposable
    {
        protected bool disposed = false;
        public readonly IntPtr m_cptr;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposed) return;
            Native.dv_destroy(m_cptr);
            disposed = true;
        }

        public string name_view_cls()
        {
            return Native.dv_name_view_cls(m_cptr);
        }

        public DeviceViewable(IntPtr cptr)
        {
            m_cptr = cptr;
        }

        ~DeviceViewable()
        {
            Dispose(false);
        }

    }

    public class DVInt8 : DeviceViewable
    {
        public DVInt8(sbyte v) : base(Native.dvint8_create(v)) {}
        public sbyte value()
        {
            return Native.dvint8_value(m_cptr);
        }
    }

    public class DVUInt8 : DeviceViewable
    {
        public DVUInt8(byte v) : base(Native.dvuint8_create(v)) { }
        public byte value()
        {
            return Native.dvuint8_value(m_cptr);
        }
    }

    public class DVInt16 : DeviceViewable
    {
        public DVInt16(short v) : base(Native.dvint16_create(v)) { }
        public short value()
        {
            return Native.dvint16_value(m_cptr);
        }
    }

    public class DVUInt16 : DeviceViewable
    {
        public DVUInt16(ushort v) : base(Native.dvuint16_create(v)) { }
        public ushort value()
        {
            return Native.dvuint16_value(m_cptr);
        }
    }

    public class DVInt32 : DeviceViewable
    {
        public DVInt32(int v) : base(Native.dvint32_create(v)) { }
        public int value()
        {
            return Native.dvint32_value(m_cptr);
        }
    }

    public class DVUInt32 : DeviceViewable
    {
        public DVUInt32(uint v) : base(Native.dvuint32_create(v)) { }
        public uint value()
        {
            return Native.dvuint32_value(m_cptr);
        }
    }

    public class DVInt64 : DeviceViewable
    {
        public DVInt64(long v) : base(Native.dvint64_create(v)) { }
        public long value()
        {
            return Native.dvint64_value(m_cptr);
        }
    }

    public class DVUInt64 : DeviceViewable
    {
        public DVUInt64(ulong v) : base(Native.dvuint64_create(v)) { }
        public ulong value()
        {
            return Native.dvuint64_value(m_cptr);
        }
    }

    public class DVFloat : DeviceViewable
    {
        public DVFloat(float v) : base(Native.dvfloat_create(v)) { }
        public float value()
        {
            return Native.dvfloat_value(m_cptr);
        }
    }

    public class DVDouble : DeviceViewable
    {
        public DVDouble(double v) : base(Native.dvdouble_create(v)) { }
        public double value()
        {
            return Native.dvdouble_value(m_cptr);
        }
    }

    public class DVBool : DeviceViewable
    {
        public DVBool(bool v) : base(Native.dvbool_create(v)) { }
        public bool value()
        {
            return Native.dvbool_value(m_cptr);
        }
    }

}
