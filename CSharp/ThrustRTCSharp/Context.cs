using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCLR;
using System.Runtime.InteropServices;

namespace ThrustRTCSharp
{
    public class Context
    {
        static void set_libnvrtc_path(String path)
        {
            IntPtr p_path = Marshal.StringToBSTR(path);
            Native.set_libnvrtc_path(p_path);
        }

        IntPtr c_ptr() { return m_cptr; }

        public Context()
        {
            m_cptr = Native.context_create();
        }

        ~Context()
        {
            Native.context_destroy(m_cptr);
        }


        public void set_verbose(bool verbose)
        {
            Native.context_set_verbose(m_cptr, verbose);
        }

        public void add_include_dir(String dir)
        {
            IntPtr p_dir = Marshal.StringToBSTR(dir);
            Native.context_add_include_dir(m_cptr, p_dir);
        }

        public void add_built_in_header(String filename, String filecontent)
        {
            IntPtr p_filename = Marshal.StringToBSTR(filename);
            IntPtr p_filecontent = Marshal.StringToBSTR(filecontent);
            Native.context_add_built_in_header(m_cptr, p_filename, p_filecontent);
        }

        public void add_inlcude_filename(String filename)
        {
            IntPtr p_filename = Marshal.StringToBSTR(filename);
            Native.context_add_include_filename(m_cptr, p_filename);
        }

        public void add_code_block(String code)
        {
            IntPtr p_code = Marshal.StringToBSTR(code);
            Native.context_add_code_block(m_cptr, p_code);
        }

        private IntPtr m_cptr;
    }
}
