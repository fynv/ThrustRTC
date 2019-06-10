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
		static IntPtr context_create();
		static void context_destroy(IntPtr p_ctx);
		static void context_set_verbose(IntPtr p_ctx, bool verbose);
		static void context_add_include_dir(IntPtr p_ctx, IntPtr p_dir);
		static void context_add_built_in_header(IntPtr p_ctx, IntPtr p_filename, IntPtr p_filecontent);
		static void context_add_include_filename(IntPtr p_ctx, IntPtr p_filename);
		static void context_add_code_block(IntPtr p_ctx, IntPtr p_code);

		// DeviceViewable


	};
}

