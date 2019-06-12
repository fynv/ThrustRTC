#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "TRTCContext.h"

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

	IntPtr Native::kernel_create(array<IntPtr>^ p_param_names, IntPtr p_code_body)
	{
		int num_params = p_param_names->Length;
		std::vector<const char*> params(num_params);
		for (int i = 0; i < num_params; i++)
			params[i] = just_cast_it<const char>(p_param_names[i]);
		const char* body = just_cast_it<const char>(p_code_body);
		TRTC_Kernel* cptr = new TRTC_Kernel(params, body);
		return IntPtr(cptr);
	}

	void Native::kernel_destroy(IntPtr p_kernel)
	{
		TRTC_Kernel* kernel = just_cast_it<TRTC_Kernel>(p_kernel);
		delete kernel;
	}

	int Native::kernel_num_params(IntPtr p_kernel)
	{
		TRTC_Kernel* kernel = just_cast_it<TRTC_Kernel>(p_kernel);
		return (int)kernel->num_params();
	}

	int Native::kernel_calc_optimal_block_size(IntPtr p_kernel, IntPtr p_args, uint32_t sharedMemBytes)
	{
		TRTC_Kernel* kernel = just_cast_it<TRTC_Kernel>(p_kernel);
		const DeviceViewable** args = just_cast_it<const DeviceViewable*>(p_args);
		int sizeBlock;
		if (kernel->calc_optimal_block_size(args, sizeBlock, sharedMemBytes))
			return sizeBlock;
		else
			return -1;
	}

	int Native::kernel_calc_number_blocks(IntPtr p_kernel, IntPtr p_args, int sizeBlock, uint32_t sharedMemBytes)
	{
		TRTC_Kernel* kernel = just_cast_it<TRTC_Kernel>(p_kernel);
		const DeviceViewable** args = just_cast_it<const DeviceViewable*>(p_args);
		int numBlocks;
		if (kernel->calc_number_blocks(args, sizeBlock, numBlocks, sharedMemBytes))
			return numBlocks;
		else
			return -1;
	}

	static dim_type to_cpp(dim_type_clr in)
	{
		return { in.x, in.y, in.z };
	}

	bool Native::kernel_launch(IntPtr p_kernel, dim_type_clr gridDim, dim_type_clr blockDim, IntPtr p_args, uint32_t sharedMemBytes)
	{
		TRTC_Kernel* kernel = just_cast_it<TRTC_Kernel>(p_kernel);
		const DeviceViewable** args = just_cast_it<const DeviceViewable*>(p_args);
		return kernel->launch(to_cpp(gridDim), to_cpp(blockDim), args, sharedMemBytes);
	}

	IntPtr Native::for_create(array<IntPtr>^ param_names, IntPtr p_name_iter, IntPtr p_code_body)
	{
		int num_params = param_names->Length;
		std::vector<const char*> params(num_params);
		for (int i = 0; i < num_params; i++)
			params[i] = just_cast_it<const char>(param_names[i]);
		const char* idx = just_cast_it<const char>(p_name_iter);
		const char* body = just_cast_it<const char>(p_code_body);
		TRTC_For* cptr = new TRTC_For(params, idx, body);
		return IntPtr(cptr);
	}

	void Native::for_destroy(IntPtr p_kernel)
	{
		TRTC_For* kernel = just_cast_it<TRTC_For>(p_kernel);
		delete kernel;
	}

	int Native::for_num_params(IntPtr p_kernel)
	{
		TRTC_For* kernel = just_cast_it<TRTC_For>(p_kernel);
		return (int)kernel->num_params();
	}

	bool Native::for_launch(IntPtr p_kernel, size_t begin, size_t end, IntPtr p_args)
	{
		TRTC_For* kernel = just_cast_it<TRTC_For>(p_kernel);
		const DeviceViewable** args = just_cast_it<const DeviceViewable*>(p_args);
		return kernel->launch(begin, end, args);
	}

}

