using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCLR;
using System.Runtime.InteropServices;

namespace ThrustRTCSharp
{
    public class DVVectorLike : DeviceViewable
    {
        public DVVectorLike(IntPtr cptr) : base(cptr) {}

        public String name_elem_cls()
        {
            return Native.dvvectorlike_name_elem_cls(m_cptr);
        }

        public ulong size()
        {
            return Native.dvvectorlike_size(m_cptr);
        }
    }

    public class DVVector : DVVectorLike
    {
        public DVVector(String elem_cls, ulong size, IntPtr hdata = default(IntPtr))
            : base(Native.dvvector_create(Marshal.StringToHGlobalAnsi(elem_cls), size, hdata)) {}

        public DVVector(sbyte[] hdata) : this("int8_t", (ulong)hdata.Length, Marshal.UnsafeAddrOfPinnedArrayElement(hdata, 0)) { }
        public DVVector(byte[] hdata) : this("uint8_t", (ulong)hdata.Length, Marshal.UnsafeAddrOfPinnedArrayElement(hdata, 0)) { }
        public DVVector(short[] hdata) : this("int16_t", (ulong)hdata.Length, Marshal.UnsafeAddrOfPinnedArrayElement(hdata, 0)) { }
        public DVVector(ushort[] hdata) : this("uint16_t", (ulong)hdata.Length, Marshal.UnsafeAddrOfPinnedArrayElement(hdata, 0)) { }
        public DVVector(int[] hdata) : this("int32_t", (ulong)hdata.Length, Marshal.UnsafeAddrOfPinnedArrayElement(hdata, 0)) { }
        public DVVector(uint[] hdata) : this("uint32_t", (ulong)hdata.Length, Marshal.UnsafeAddrOfPinnedArrayElement(hdata, 0)) { }
        public DVVector(long[] hdata) : this("int64_t", (ulong)hdata.Length, Marshal.UnsafeAddrOfPinnedArrayElement(hdata, 0)) { }
        public DVVector(ulong[] hdata) : this("uint64_t", (ulong)hdata.Length, Marshal.UnsafeAddrOfPinnedArrayElement(hdata, 0)) { }
        public DVVector(float[] hdata) : this("float", (ulong)hdata.Length, Marshal.UnsafeAddrOfPinnedArrayElement(hdata, 0)) { }
        public DVVector(double[] hdata) : this("double", (ulong)hdata.Length, Marshal.UnsafeAddrOfPinnedArrayElement(hdata, 0)) { }
        public DVVector(bool[] hdata) : this("bool", (ulong)hdata.Length, Marshal.UnsafeAddrOfPinnedArrayElement(hdata, 0)) { }

        public void to_host(IntPtr hdata, long begin = 0, long end = -1)
        {
            Native.dvvector_to_host(m_cptr, hdata, (ulong)begin, (ulong)end);
        }

        public object to_host(long begin = 0, long end = -1)
        {
            String type = name_elem_cls();
            if (end == -1) end = (long)size();
            if (end < begin) return null;
            ulong h_size = (ulong)(end - begin);

            if (type=="int8_t")
            {
                sbyte[] arr = new sbyte[h_size];
                if (h_size>0)
                {
                    to_host(Marshal.UnsafeAddrOfPinnedArrayElement(arr, 0), begin, end);
                    return arr;
                }
            }
            else if (type == "uint8_t")
            {
                byte[] arr = new byte[h_size];
                if (h_size > 0)
                {
                    to_host(Marshal.UnsafeAddrOfPinnedArrayElement(arr, 0), begin, end);
                    return arr;
                }
            }
            else if (type == "int16_t")
            {
                short[] arr = new short[h_size];
                if (h_size > 0)
                {
                    to_host(Marshal.UnsafeAddrOfPinnedArrayElement(arr, 0), begin, end);
                    return arr;
                }
            }
            else if (type == "uint16_t")
            {
                ushort[] arr = new ushort[h_size];
                if (h_size > 0)
                {
                    to_host(Marshal.UnsafeAddrOfPinnedArrayElement(arr, 0), begin, end);
                    return arr;
                }
            }
            else if (type == "int32_t")
            {
                int[] arr = new int[h_size];
                if (h_size > 0)
                {
                    to_host(Marshal.UnsafeAddrOfPinnedArrayElement(arr, 0), begin, end);
                    return arr;
                }
            }
            else if (type == "uint32_t")
            {
                uint[] arr = new uint[h_size];
                if (h_size > 0)
                {
                    to_host(Marshal.UnsafeAddrOfPinnedArrayElement(arr, 0), begin, end);
                    return arr;
                }
            }
            else if (type == "int64_t")
            {
                long[] arr = new long[h_size];
                if (h_size > 0)
                {
                    to_host(Marshal.UnsafeAddrOfPinnedArrayElement(arr, 0), begin, end);
                    return arr;
                }
            }
            else if (type == "uint64_t")
            {
                ulong[] arr = new ulong[h_size];
                if (h_size > 0)
                {
                    to_host(Marshal.UnsafeAddrOfPinnedArrayElement(arr, 0), begin, end);
                    return arr;
                }
            }
            else if (type == "float")
            {
                float[] arr = new float[h_size];
                if (h_size > 0)
                {
                    to_host(Marshal.UnsafeAddrOfPinnedArrayElement(arr, 0), begin, end);
                    return arr;
                }
            }
            else if (type == "double")
            {
                double[] arr = new double[h_size];
                if (h_size > 0)
                {
                    to_host(Marshal.UnsafeAddrOfPinnedArrayElement(arr, 0), begin, end);
                    return arr;
                }
            }
            else if (type == "bool")
            {
                bool[] arr = new bool[h_size];
                if (h_size > 0)
                {
                    to_host(Marshal.UnsafeAddrOfPinnedArrayElement(arr, 0), begin, end);
                    return arr;
                }
            }
            return null;
        }


    }
}
