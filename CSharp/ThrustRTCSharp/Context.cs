using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCLR;
using System.Runtime.InteropServices;

namespace ThrustRTCSharp
{
    public class TRTC
    {
        static void set_libnvrtc_path(String path)
        {
            IntPtr p_path = Marshal.StringToBSTR(path);
            Native.set_libnvrtc_path(p_path);
        }

        static public void Set_Verbose(bool verbose)
        {
            Native.set_verbose(verbose);
        }

        static public void Add_Include_Dir(String dir)
        {
            IntPtr p_dir = Marshal.StringToBSTR(dir);
            Native.add_include_dir(p_dir);
        }

        static public void Add_Built_In_Header(String filename, String filecontent)
        {
            IntPtr p_filename = Marshal.StringToBSTR(filename);
            IntPtr p_filecontent = Marshal.StringToBSTR(filecontent);
            Native.add_built_in_header(p_filename, p_filecontent);
        }

        public void Add_Inlcude_Filename(String filename)
        {
            IntPtr p_filename = Marshal.StringToBSTR(filename);
            Native.add_include_filename(p_filename);
        }

        static public void Add_Code_Block(String code)
        {
            IntPtr p_code = Marshal.StringToBSTR(code);
            Native.add_code_block(p_code);
        }

    }
}
