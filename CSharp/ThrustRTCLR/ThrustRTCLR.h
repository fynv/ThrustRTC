#pragma once

#include <cstdint>

using namespace System;

namespace ThrustRTCLR 
{
	public value struct dim_type_clr
	{
		unsigned int x, y, z;
	};

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

		// Kernel
		static IntPtr kernel_create(array<IntPtr>^ p_param_names, IntPtr p_code_body);
		static void kernel_destroy(IntPtr p_kernel);
		static int kernel_num_params(IntPtr p_kernel);
		static int kernel_calc_optimal_block_size(IntPtr p_kernel, IntPtr p_args, uint32_t sharedMemBytes);
		static int kernel_calc_number_blocks(IntPtr p_kernel, IntPtr p_args,  int sizeBlock, uint32_t sharedMemBytes);
		static bool kernel_launch(IntPtr p_kernel, dim_type_clr gridDim, dim_type_clr blockDim, IntPtr p_args, uint32_t sharedMemBytes);

		// For-loop
		static IntPtr for_create(array<IntPtr>^ param_names, IntPtr p_name_iter, IntPtr p_code_body);
		static void for_destroy(IntPtr p_kernel);
		static int for_num_params(IntPtr p_kernel);
		static bool for_launch(IntPtr p_kernel, size_t begin, size_t end, IntPtr p_args);
		static bool for_launch_n(IntPtr p_kernel, size_t n, IntPtr p_args);

		// DeviceViewable
		static String^ dv_name_view_cls(IntPtr p_dv);
		static void dv_destroy(IntPtr p_dv);

		// basic types
		static IntPtr dvint8_create(int8_t v);
		static int8_t dvint8_value(IntPtr p);
		static IntPtr dvuint8_create(uint8_t v);
		static uint8_t dvuint8_value(IntPtr p);
		static IntPtr dvint16_create(int16_t v);
		static int16_t dvint16_value(IntPtr p);
		static IntPtr dvuint16_create(uint16_t v);
		static uint16_t dvuint16_value(IntPtr p);
		static IntPtr dvint32_create(int32_t v);
		static int32_t dvint32_value(IntPtr p);
		static IntPtr dvuint32_create(uint32_t v);
		static uint32_t dvuint32_value(IntPtr p);
		static IntPtr dvint64_create(int64_t v);
		static int64_t dvint64_value(IntPtr p);
		static IntPtr dvuint64_create(uint64_t v);
		static uint64_t dvuint64_value(IntPtr p);
		static IntPtr dvfloat_create(float v);
		static float dvfloat_value(IntPtr p);
		static IntPtr dvdouble_create(double v);
		static double dvdouble_value(IntPtr p);
		static IntPtr dvbool_create(bool v);
		static bool dvbool_value(IntPtr p);

		// DVVectorLike
		static String^ dvvectorlike_name_elem_cls(IntPtr p_dvvec);
		static size_t dvvectorlike_size(IntPtr p_dvvec);
		
		// DVVector
		static IntPtr dvvector_create(IntPtr p_elem_cls, size_t size, IntPtr p_hdata);
		static void dvvector_to_host(IntPtr p_dvvec, IntPtr p_hdata, size_t begin, size_t end);

	};
}

