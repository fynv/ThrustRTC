#pragma once

using namespace System;

#include "TRTCContext.h"

namespace ThrustRTCLR 
{
	public ref class Native
	{
	public:
		static void set_libnvrtc_path(IntPtr p_path);

		// Context
		static void set_verbose(bool verbose);
		static void add_include_dir(IntPtr p_dir);
		static void add_built_in_header(IntPtr p_filename, IntPtr p_filecontent);
		static void add_include_filename(IntPtr p_filename);
		static void add_code_block(IntPtr p_code);

		// DeviceViewable


	};
}

