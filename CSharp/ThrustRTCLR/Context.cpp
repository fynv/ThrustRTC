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
		TRTCContext::set_libnvrtc_path(c_path);
	}

	IntPtr Native::context_create()
	{
		return IntPtr(new TRTCContext());
	}

	void Native::context_destroy(IntPtr p_ctx)
	{
		TRTCContext* c_ctx = just_cast_it<TRTCContext>(p_ctx);
		delete c_ctx;
	}

	void Native::context_set_verbose(IntPtr p_ctx, bool verbose)
	{
		TRTCContext* c_ctx = just_cast_it<TRTCContext>(p_ctx);
		c_ctx->set_verbose(verbose);
	}

	void Native::context_add_include_dir(IntPtr p_ctx, IntPtr p_dir)
	{
		TRTCContext* c_ctx = just_cast_it<TRTCContext>(p_ctx);
		const char* c_dir = just_cast_it<const char>(p_dir);
		c_ctx->add_include_dir(c_dir);
	}

	void Native::context_add_built_in_header(IntPtr p_ctx, IntPtr p_filename, IntPtr p_filecontent)
	{
		TRTCContext* c_ctx = just_cast_it<TRTCContext>(p_ctx);
		const char* c_filename = just_cast_it<const char>(p_filename);
		const char* c_filecontent = just_cast_it<const char>(p_filecontent);
		c_ctx->add_built_in_header(c_filename, c_filecontent);
	}

	void Native::context_add_include_filename(IntPtr p_ctx, IntPtr p_filename)
	{
		TRTCContext* c_ctx = just_cast_it<TRTCContext>(p_ctx);
		const char* c_filename = just_cast_it<const char>(p_filename);
		c_ctx->add_inlcude_filename(c_filename);
	}

	void Native::context_add_code_block(IntPtr p_ctx, IntPtr p_code)
	{
		TRTCContext* c_ctx = just_cast_it<TRTCContext>(p_ctx);
		const char* c_code = just_cast_it<const char>(p_code);
		c_ctx->add_code_block(c_code);
	}
}

