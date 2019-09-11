using System;
using ThrustRTCLR;
using System.Runtime.InteropServices;

namespace ThrustRTCSharp
{
    public struct dim_type
    {
        public uint x;
        public uint y;
        public uint z;

        public static implicit operator dim_type(uint x)
        {
            return new dim_type { x = x, y = 1, z = 1 };
        }
    };

    public partial class TRTC
    {
        static void set_libnvrtc_path(string path)
        {
            IntPtr p_path = Marshal.StringToHGlobalAnsi(path);
            Native.set_libnvrtc_path(p_path);
        }

        public static void Set_Verbose(bool verbose = true)
        {
            Native.set_verbose(verbose);
        }

        public static void Add_Include_Dir(string dir)
        {
            IntPtr p_dir = Marshal.StringToHGlobalAnsi(dir);
            Native.add_include_dir(p_dir);
        }

        public static void Add_Built_In_Header(string filename, string filecontent)
        {
            IntPtr p_filename = Marshal.StringToHGlobalAnsi(filename);
            IntPtr p_filecontent = Marshal.StringToHGlobalAnsi(filecontent);
            Native.add_built_in_header(p_filename, p_filecontent);
        }

        public static void Add_Inlcude_Filename(string filename)
        {
            IntPtr p_filename = Marshal.StringToHGlobalAnsi(filename);
            Native.add_include_filename(p_filename);
        }

        public static void Add_Code_Block(string code)
        {
            IntPtr p_code = Marshal.StringToHGlobalAnsi(code);
            Native.add_code_block(p_code);
        }

        public static void Wait()
        {
            Native.wait();
        }
    }

    partial class Internal
    {
        public static IntPtr[] ConvertStrList(string[] strs)
        {
            IntPtr[] p_strs = new IntPtr[strs.Length];
            for (int i = 0; i < strs.Length; i++)
                p_strs[i] = Marshal.StringToHGlobalAnsi(strs[i]);
            return p_strs;
        }

        public static IntPtr ConvertDVList(DeviceViewable[] args)
        {
            if (args.Length > 0)
            {
                IntPtr[] p_args = new IntPtr[args.Length];
                for (int i = 0; i < args.Length; i++)
                    p_args[i] = args[i].m_cptr;
                return Marshal.UnsafeAddrOfPinnedArrayElement(p_args, 0);
            }
            else
            {
                return IntPtr.Zero;
            }
        }

        public static dim_type_clr ConvertDimType(dim_type dim)
        {
            return new dim_type_clr { x = dim.x, y = dim.y, z = dim.z };
        }
    }

    public class Kernel
    {
        public readonly IntPtr m_cptr;

        public Kernel(string[] param_names, string body)
        {
            IntPtr[] p_param_names = Internal.ConvertStrList(param_names);
            IntPtr p_body = Marshal.StringToHGlobalAnsi(body);
            m_cptr = Native.kernel_create(p_param_names, p_body);
        }

        ~Kernel()
        {
            Native.kernel_destroy(m_cptr);
        }

        public int num_params()
        {
            return Native.kernel_num_params(m_cptr);
        }

        public int calc_optimal_block_size(DeviceViewable[] args, uint sharedMemBytes = 0)
        {
            IntPtr p_args = Internal.ConvertDVList(args);
            return Native.kernel_calc_optimal_block_size(m_cptr, p_args, sharedMemBytes);
        }

        public int calc_number_blocks(DeviceViewable[] args, int sizeBlock, uint sharedMemBytes = 0)
        {
            IntPtr p_args = Internal.ConvertDVList(args);
            return Native.kernel_calc_number_blocks(m_cptr, p_args, sizeBlock, sharedMemBytes);
        }

        public bool launch(dim_type gridDim, dim_type blockDim, DeviceViewable[] args, uint sharedMemBytes = 0)
        {
            IntPtr p_args = Internal.ConvertDVList(args);
            return Native.kernel_launch(m_cptr, Internal.ConvertDimType(gridDim), Internal.ConvertDimType(blockDim), p_args, sharedMemBytes);
        }
    }

    public class For
    {
        public readonly IntPtr m_cptr;

        public For(string[] param_names, string name_iter, string body)
        {
            IntPtr[] p_param_names = Internal.ConvertStrList(param_names);
            IntPtr p_name_iter = Marshal.StringToHGlobalAnsi(name_iter);
            IntPtr p_body = Marshal.StringToHGlobalAnsi(body);
            m_cptr = Native.for_create(p_param_names, p_name_iter, p_body);
        }
        
        ~For()
        {
            Native.for_destroy(m_cptr);
        }

        public int num_params()
        {
            return Native.for_num_params(m_cptr);
        }

        public bool launch(ulong begin, ulong end, DeviceViewable[] args)
        {
            IntPtr p_args = Internal.ConvertDVList(args);
            return Native.for_launch(m_cptr, begin, end, p_args);
        }

        public bool launch_n(ulong n, DeviceViewable[] args)
        {
            IntPtr p_args = Internal.ConvertDVList(args);
            return Native.for_launch_n(m_cptr, n, p_args);
        }
    }
}
