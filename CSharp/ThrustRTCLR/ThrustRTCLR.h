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
		static int kernel_calc_number_blocks(IntPtr p_kernel, IntPtr p_args, int sizeBlock, uint32_t sharedMemBytes);
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
		static bool tabulate(IntPtr p_vec, IntPtr p_op, size_t begin, size_t end);
		static bool transform(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_op, size_t begin_in, size_t end_in, size_t begin_out);
		static bool transform_binary(IntPtr p_vec_in1, IntPtr p_vec_in2, IntPtr p_vec_out, IntPtr p_op, size_t begin_in1, size_t end_in1, size_t begin_in2, size_t begin_out);
		static bool transform_if(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_op, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_out);
		static bool transform_if_stencil(IntPtr p_vec_in, IntPtr p_vec_stencil, IntPtr p_vec_out, IntPtr p_op, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out);
		static bool transform_binary_if_stencil(IntPtr p_vec_in1, IntPtr p_vec_in2, IntPtr p_vec_stencil, IntPtr p_vec_out, IntPtr p_op, IntPtr p_pred, size_t begin_in1, size_t end_in1, size_t begin_in2, size_t begin_stencil, size_t begin_out);

		// Copying
		static bool gather(IntPtr p_vec_map, IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_map, size_t end_map, size_t begin_in, size_t begin_out);
		static bool gather_if(IntPtr p_vec_map, IntPtr p_vec_stencil, IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_map, size_t end_map, size_t begin_stencil, size_t begin_in, size_t begin_out);
		static bool gather_if(IntPtr p_vec_map, IntPtr p_vec_stencil, IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_pred, size_t begin_map, size_t end_map, size_t begin_stencil, size_t begin_in, size_t begin_out);
		static bool scatter(IntPtr p_vec_in, IntPtr p_vec_map, IntPtr p_vec_out, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_out);
		static bool scatter_if(IntPtr p_vec_in, IntPtr p_vec_map, IntPtr p_vec_stencil, IntPtr p_vec_out, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_stencil, size_t begin_out);
		static bool scatter_if(IntPtr p_vec_in, IntPtr p_vec_map, IntPtr p_vec_stencil, IntPtr p_vec_out, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_stencil, size_t begin_out);
		static bool copy(IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_in, size_t end_in, size_t begin_out);
		static bool swap(IntPtr p_vec1, IntPtr p_vec2, size_t begin1, size_t end1, size_t begin2);

		// Redutions
		static size_t count(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end);
		static size_t count_if(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end);
		static Object^ reduce(IntPtr p_vec, size_t begin, size_t end);
		static Object^ reduce(IntPtr p_vec, IntPtr p_init, size_t begin, size_t end);
		static Object^ reduce(IntPtr p_vec, IntPtr p_init, IntPtr p_binary_op, size_t begin, size_t end);
		static uint32_t reduce_by_key(IntPtr p_key_in, IntPtr p_value_in, IntPtr p_key_out, IntPtr p_value_out, size_t begin_key_in, size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out);
		static uint32_t reduce_by_key(IntPtr p_key_in, IntPtr p_value_in, IntPtr p_key_out, IntPtr p_value_out, IntPtr p_binary_pred, size_t begin_key_in, size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out);
		static uint32_t reduce_by_key(IntPtr p_key_in, IntPtr p_value_in, IntPtr p_key_out, IntPtr p_value_out, IntPtr p_binary_pred, IntPtr p_binary_op, size_t begin_key_in, size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out);
		static Object^ equal(IntPtr p_vec1, IntPtr p_vec2, size_t begin1, size_t end1, size_t begin2);
		static Object^ equal(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_binary_pred, size_t begin1, size_t end1, size_t begin2);
		static size_t min_element(IntPtr p_vec, size_t begin, size_t end);
		static size_t min_element(IntPtr p_vec, IntPtr p_comp, size_t begin, size_t end);
		static size_t max_element(IntPtr p_vec, size_t begin, size_t end);
		static size_t max_element(IntPtr p_vec, IntPtr p_comp, size_t begin, size_t end);
		static Tuple<int64_t, int64_t>^ minmax_element(IntPtr p_vec, size_t begin, size_t end);
		static Tuple<int64_t, int64_t>^ minmax_element(IntPtr p_vec, IntPtr p_comp, size_t begin, size_t end);
		static Object^ inner_product(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_init, size_t begin1, size_t end1, size_t begin2);
		static Object^ inner_product(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_init, IntPtr p_binary_op1, IntPtr p_binary_op2, size_t begin1, size_t end1, size_t begin2);
		static Object^ transform_reduce(IntPtr p_vec, IntPtr p_unary_op, IntPtr p_init, IntPtr p_binary_op, size_t begin, size_t end);
		static Object^ all_of(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end);
		static Object^ any_of(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end);
		static Object^ none_of(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end);
		static Object^ is_partitioned(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end);
		static Object^ is_sorted(IntPtr p_vec, size_t begin, size_t end);
		static Object^ is_sorted(IntPtr p_vec, IntPtr p_comp, size_t begin, size_t end);

		// PrefixSums
		static bool inclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_in, size_t end_in, size_t begin_out);
		static bool inclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_binary_op, size_t begin_in, size_t end_in, size_t begin_out);
		static bool exclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_in, size_t end_in, size_t begin_out);
		static bool exclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_init, size_t begin_in, size_t end_in, size_t begin_out);
		static bool exclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_init, IntPtr p_binary_op, size_t begin_in, size_t end_in, size_t begin_out);
		static bool inclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out);
		static bool inclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, IntPtr p_binary_pred, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out);
		static bool inclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, IntPtr p_binary_pred, IntPtr p_binary_op, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out);
		static bool exclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out);
		static bool exclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, IntPtr p_init, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out);
		static bool exclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, IntPtr p_init, IntPtr p_binary_pred, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out);
		static bool exclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, IntPtr p_init, IntPtr p_binary_pred, IntPtr p_binary_op, size_t begin_key, size_t end_key, size_t begin_value, size_t begin_out);
		static bool transform_inclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_unary_op, IntPtr p_binary_op, size_t begin_in, size_t end_in, size_t begin_out);
		static bool transform_exclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_unary_op, IntPtr p_init, IntPtr p_binary_op, size_t begin_in, size_t end_in, size_t begin_out);

		// Reordering
		static uint32_t copy_if(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_out);
		static uint32_t copy_if_stencil(IntPtr p_vec_in, IntPtr p_vec_stencil, IntPtr p_vec_out, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out);
		static uint32_t remove(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end);
		static uint32_t remove_copy(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_value, size_t begin_in, size_t end_in, size_t begin_out);
		static uint32_t remove_if(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end);
		static uint32_t remove_copy_if(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_out);
		static uint32_t remove_if_stencil(IntPtr p_vec, IntPtr p_stencil, IntPtr p_pred, size_t begin, size_t end, size_t begin_stencil);
		static uint32_t remove_copy_if_stencil(IntPtr p_vec_in, IntPtr p_stencil, IntPtr p_vec_out, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out);
		static uint32_t unique(IntPtr p_vec, size_t begin, size_t end);
		static uint32_t unique(IntPtr p_vec, IntPtr p_binary_pred, size_t begin, size_t end);
		static uint32_t unique_copy(IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_in, size_t end_in, size_t begin_out);
		static uint32_t unique_copy(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_binary_pred, size_t begin_in, size_t end_in, size_t begin_out);
		static uint32_t unique_by_key(IntPtr p_keys, IntPtr p_values, size_t begin_key, size_t end_key, size_t begin_value);
		static uint32_t unique_by_key(IntPtr p_keys, IntPtr p_values, IntPtr p_binary_pred, size_t begin_key, size_t end_key, size_t begin_value);
		static uint32_t unique_by_key_copy(IntPtr p_keys_in, IntPtr p_values_in, IntPtr p_keys_out, IntPtr p_values_out, size_t begin_key_in, size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out);
		static uint32_t unique_by_key_copy(IntPtr p_keys_in, IntPtr p_values_in, IntPtr p_keys_out, IntPtr p_values_out, IntPtr p_binary_pred, size_t begin_key_in, size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out);
		static uint32_t partition(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end);
		static uint32_t partition_stencil(IntPtr p_vec, IntPtr p_stencil, IntPtr p_pred, size_t begin, size_t end, size_t begin_stencil);
		static uint32_t partition_copy(IntPtr p_vec_in, IntPtr p_vec_true, IntPtr p_vec_false, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_true, size_t begin_false);
		static uint32_t partition_copy_stencil(IntPtr p_vec_in, IntPtr p_stencil, IntPtr p_vec_true, IntPtr p_vec_false, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_true, size_t begin_false);

		// Searching
		static Object^ find(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end);
		static Object^ find_if(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end);
		static Object^ find_if_not(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end);
		static Tuple<int64_t, int64_t>^ mismatch(IntPtr p_vec1, IntPtr p_vec2, size_t begin1, size_t end1, size_t begin2);
		static Tuple<int64_t, int64_t>^ mismatch(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_pred, size_t begin1, size_t end1, size_t begin2);
		static Object^ lower_bound(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end);
		static Object^ lower_bound(IntPtr p_vec, IntPtr p_value, IntPtr p_comp, size_t begin, size_t end);
		static Object^ upper_bound(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end);
		static Object^ upper_bound(IntPtr p_vec, IntPtr p_value, IntPtr p_comp, size_t begin, size_t end);
		static Object^ binary_search(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end);
		static Object^ binary_search(IntPtr p_vec, IntPtr p_value, IntPtr p_comp, size_t begin, size_t end);
		static bool lower_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result);
		static bool lower_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, IntPtr p_comp, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result);
		static bool upper_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result);
		static bool upper_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, IntPtr p_comp, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result);
		static bool binary_search_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result);
		static bool binary_search_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, IntPtr p_comp, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result);
		static Object^ partition_point(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end);
		static Object^ is_sorted_until(IntPtr p_vec, size_t begin, size_t end);
		static Object^ is_sorted_until(IntPtr p_vec, IntPtr p_comp, size_t begin, size_t end);

		// Merging
		static bool merge(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_vec_out, size_t begin1, size_t end1, size_t begin2, size_t end2, size_t begin_out);
		static bool merge(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_vec_out, IntPtr p_comp, size_t begin1, size_t end1, size_t begin2, size_t end2, size_t begin_out);
		static bool merge_by_key(IntPtr p_key1, IntPtr p_keys2, IntPtr p_value1, IntPtr p_value2, IntPtr p_keys_out, IntPtr p_value_out, size_t begin_keys1, size_t end_keys1, size_t begin_keys2, size_t end_keys2, size_t begin_value1, size_t begin_value2, size_t begin_keys_out, size_t begin_value_out);
		static bool merge_by_key(IntPtr p_key1, IntPtr p_keys2, IntPtr p_value1, IntPtr p_value2, IntPtr p_keys_out, IntPtr p_value_out, IntPtr p_comp, size_t begin_keys1, size_t end_keys1, size_t begin_keys2, size_t end_keys2, size_t begin_value1, size_t begin_value2, size_t begin_keys_out, size_t begin_value_out);

		// Sorting
		static bool sort(IntPtr p_vec, size_t begin, size_t end);
		static bool sort(IntPtr p_vec, IntPtr p_comp, size_t begin, size_t end);
		static bool sort_by_key(IntPtr p_keys, IntPtr p_values, size_t begin_keys, size_t end_keys, size_t begin_values);
		static bool sort_by_key(IntPtr p_keys, IntPtr p_values, IntPtr p_comp, size_t begin_keys, size_t end_keys, size_t begin_values);

	};
}

