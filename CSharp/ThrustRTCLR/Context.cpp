#include "stdafx.h"
#include "ThrustRTCLR.h"

namespace ThrustRTCLR 
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	void Native::set_libnvrtc_path(IntPtr p_path)
	{
		const char* c_path = just_cast_it<const char>(p_path);
		::set_libnvrtc_path(c_path);
	}

	void Native::set_verbose(bool verbose)
	{
		TRTC_Set_Verbose(verbose);
	}

	void Native::add_include_dir(IntPtr p_dir)
	{
		const char* c_dir = just_cast_it<const char>(p_dir);
		TRTC_Add_Include_Dir(c_dir);
	}

	void Native::add_built_in_header(IntPtr p_filename, IntPtr p_filecontent)
	{
		const char* c_filename = just_cast_it<const char>(p_filename);
		const char* c_filecontent = just_cast_it<const char>(p_filecontent);
		TRTC_Add_Built_In_Header(c_filename, c_filecontent);
	}

	void Native::add_include_filename(IntPtr p_filename)
	{
		const char* c_filename = just_cast_it<const char>(p_filename);
		TRTC_Add_Inlcude_Filename(c_filename);
	}

	void Native::add_code_block(IntPtr p_code)
	{
		const char* c_code = just_cast_it<const char>(p_code);
		TRTC_Add_Code_Block(c_code);
	}
}

