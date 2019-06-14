#pragma once

#include <cstdint>

using namespace System;

namespace ThrustRTCLR 
{
	public value struct dim_type_clr
	{
		unsigned int x, y, z;
	};

	public value struct CapturedDeviceViewable_clr
	{
		IntPtr obj_name;
		IntPtr obj;
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
		static IntPtr for_create(array<IntPtr>^ p_param_names, IntPtr p_name_iter, IntPtr p_code_body);
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

		// DVVector
		static String^ dvvectorlike_name_elem_cls(IntPtr p_dvvec);
		static size_t dvvectorlike_size(IntPtr p_dvvec);
		static IntPtr dvvector_create(IntPtr p_elem_cls, size_t size, IntPtr p_hdata);
		static void dvvector_to_host(IntPtr p_dvvec, IntPtr p_hdata, size_t begin, size_t end);

		// Tuple
		static IntPtr dvtuple_create(array<CapturedDeviceViewable_clr>^ p_elem_map);

		// Fake-Vectors
		static IntPtr dvconstant_create(IntPtr p_dvobj, size_t size);
		static IntPtr dvcounter_create(IntPtr p_dvobj_init, size_t size);
		static IntPtr dvdiscard_create(IntPtr p_elem_cls, size_t size);
		static IntPtr dvpermutation_create(IntPtr p_vec_value, IntPtr p_vec_index);
		static IntPtr dvreverse_create(IntPtr p_vec_value);
		static IntPtr dvtransform_create(IntPtr p_vec_in, IntPtr p_elem_cls, IntPtr p_op);
		static IntPtr dvzipped_create(array<IntPtr>^ p_vecs, array<IntPtr>^ p_elem_names);
		static IntPtr dvcustomvector_create(array<CapturedDeviceViewable_clr>^ p_arg_map, IntPtr p_name_idx, IntPtr p_code_body, IntPtr p_elem_cls, size_t size, bool read_only);

		// Functor
		static IntPtr functor_create(array<CapturedDeviceViewable_clr>^ p_arg_map, array<IntPtr>^ p_functor_params, IntPtr p_code_body);
		static IntPtr built_in_functor_create(IntPtr p_name_built_in_view_cls);

		// Transformations
		static bool fiil(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end);
		static bool replace(IntPtr p_vec, IntPtr p_old_value, IntPtr p_new_value, size_t begin, size_t end);
		static bool replace_if(IntPtr p_vec, IntPtr p_pred, IntPtr p_new_value, size_t begin, size_t end);
		static bool replace_copy(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_old_value, IntPtr p_new_value, size_t begin_in, size_t end_in, size_t begin_out);
		static bool replace_copy_if(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_pred, IntPtr p_new_value, size_t begin_in, size_t end_in, size_t begin_out);
		static bool for_each(IntPtr p_vec, IntPtr p_f, size_t begin, size_t end);
		static bool adjacent_difference(IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_in, size_t end_in, size_t begin_out);
		static bool adjacent_difference(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_binary_op, size_t begin_in, size_t end_in, size_t begin_out);
		static bool sequence(IntPtr p_vec, size_t begin, size_t end);
		static bool sequence(IntPtr p_vec, IntPtr p_value_init, size_t begin, size_t end);
		static bool sequence(IntPtr p_vec, IntPtr p_value_init, IntPtr p_value_step, size_t begin, size_t end);
		
		// Copying
		static bool copy(IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_in, size_t end_in, size_t begin_out);

	};
}

