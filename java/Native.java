package JThrustRTC;

class Native 
{
	static 
	{
		System.loadLibrary("JThrustRTC_Native"); 
	}

	// Context
	public static native void set_libnvrtc_path(String path);
	public static native void set_verbose(boolean verbose);
	public static native void add_include_dir(String dir);
	public static native void add_built_in_header(String filename, String filecontent);
	public static native void add_include_filename(String filename);
	public static native void add_code_block(String code);

	public static native long kernel_create(String[] param_names, String body);
	public static native void kernel_destroy(long p_kernel);
	public static native int kernel_num_params(long p_kernel);
	public static native int kernel_calc_optimal_block_size(long p_kernel, long[] p_args, int sharedMemBytes);
	public static native int kernel_calc_number_blocks(long p_kernel, long[] p_args, int sizeBlock, int sharedMemBytes);
	public static native boolean kernel_launch(long p_kernel, int[] gridDim, int[] blockDim, long[] p_args, int sharedMemBytes);

	public static native long for_create(String[] param_names, String name_iter, String body);
	public static native void for_destroy(long p_kernel);
	public static native int for_num_params(long p_kernel);
	public static native boolean for_launch(long p_kernel, int begin, int end, long[] p_args);
	public static native boolean for_launch_n(long p_kernel, int n, long[] p_args);

	// DeviceViewable
	public static native String dv_name_view_cls(long p_dv);
	public static native void dv_destroy(long p_dv);

	public static native long dvint8_create(byte v);
	public static native byte dvint8_value(long p);
	public static native long dvint16_create(short v);
	public static native short dvint16_value(long p);
	public static native long dvint32_create(int v);
	public static native int dvint32_value(long p);
	public static native long dvint64_create(long v);
	public static native long dvint64_value(long p);
	public static native long dvfloat_create(float v);
	public static native float dvfloat_value(long p);
	public static native long dvdouble_create(double v);
	public static native double dvdouble_value(long p);

	// DVVector
	public static native String dvvectorlike_name_elem_cls(long p_dvvec);
	public static native int dvvectorlike_size(long p_dvvec);
	public static native long dvrange_create(long p_vec_value, int begin, int end);
	public static native long dvvector_create(String elem_cls, int size);
	public static native long dvvector_create(byte[] hdata);
	public static native long dvvector_create(short[] hdata);
	public static native long dvvector_create(int[] hdata);
	public static native long dvvector_create(long[] hdata);
	public static native long dvvector_create(float[] hdata);
	public static native long dvvector_create(double[] hdata);
	public static native void dvvector_to_host(long p_dvvec, byte[] hdata, int begin, int end);
	public static native void dvvector_to_host(long p_dvvec, short[] hdata, int begin, int end);
	public static native void dvvector_to_host(long p_dvvec, int[] hdata, int begin, int end);
	public static native void dvvector_to_host(long p_dvvec, long[] hdata, int begin, int end);
	public static native void dvvector_to_host(long p_dvvec, float[] hdata, int begin, int end);
	public static native void dvvector_to_host(long p_dvvec, double[] hdata, int begin, int end);

	// Tuple
	public static native long dvtuple_create(long[] p_objs, String[] name_objs);

	// Fake-Vectors
	public static native long dvconstant_create(long p_dvobj, int size);
	public static native long dvcounter_create(long p_dvobj_init, int size);
	public static native long dvdiscard_create(String elem_cls, int size);
	public static native long dvpermutation_create(long p_vec_value, long p_vec_index);
	public static native long dvreverse_create(long p_vec_value);
	public static native long dvtransform_create(long p_vec_in, String elem_cls, long p_op);
	public static native long dvzipped_create(long[] p_vecs, String[] elem_names);
	public static native long dvcustomvector_create(long[] p_objs, String[] name_objs, String name_idx, String code_body, String elem_cls, int size, boolean read_only);

	// Functor
	public static native long functor_create(String[] functor_params, String code_body);
	public static native long functor_create(long[] p_objs, String[] name_objs, String[] functor_params, String code_body);
	public static native long built_in_functor_create(String name_built_in_view_cls);

	// Transformations
	public static native boolean fill(long p_vec, long p_value);
	public static native boolean replace(long p_vec, long p_old_value, long p_new_value);
	public static native boolean replace_if(long p_vec, long p_pred, long p_new_value);
	public static native boolean replace_copy(long p_vec_in, long p_vec_out, long p_old_value, long p_new_value);
	public static native boolean replace_copy_if(long p_vec_in, long p_vec_out, long p_pred, long p_new_value);
	public static native boolean for_each(long p_vec, long p_f);
	public static native boolean adjacent_difference(long p_vec_in, long p_vec_out);
	public static native boolean adjacent_difference(long p_vec_in, long p_vec_out, long p_binary_op);
	public static native boolean sequence(long p_vec);
	public static native boolean sequence(long p_vec, long p_value_init);
	public static native boolean sequence(long p_vec, long p_value_init, long p_value_step);
	public static native boolean tabulate(long p_vec, long p_op);	
	public static native boolean transform(long p_vec_in, long p_vec_out, long p_op);
	public static native boolean transform_binary(long p_vec_in1, long p_vec_in2, long p_vec_out, long p_op);
	public static native boolean transform_if(long p_vec_in, long p_vec_out, long p_op, long p_pred);
	public static native boolean transform_if_stencil(long p_vec_in, long p_vec_stencil, long p_vec_out, long p_op, long p_pred);
	public static native boolean transform_binary_if_stencil(long p_vec_in1, long p_vec_in2, long p_vec_stencil, long p_vec_out, long p_op, long p_pred);

	// Copying
	public static native boolean gather(long p_vec_map, long p_vec_in, long p_vec_out);
	public static native boolean gather_if(long p_vec_map, long p_vec_stencil, long p_vec_in, long p_vec_out);
	public static native boolean gather_if(long p_vec_map, long p_vec_stencil, long p_vec_in, long p_vec_out, long p_pred);
	public static native boolean scatter(long p_vec_in, long p_vec_map, long p_vec_out);
	public static native boolean scatter_if(long p_vec_in, long p_vec_map, long p_vec_stencil, long p_vec_out);
	public static native boolean scatter_if(long p_vec_in, long p_vec_map, long p_vec_stencil, long p_vec_out, long p_pred);
	public static native boolean copy(long p_vec_in, long p_vec_out);
	public static native boolean swap(long p_vec1, long p_vec2);

	// Redutions
	public static native int count(long p_vec, long p_value);
	public static native int count_if(long p_vec, long p_pred);
	public static native Object reduce(long p_vec);
	public static native Object reduce(long p_vec, long p_init);
	public static native Object reduce(long p_vec, long p_init, long p_binary_op);
	public static native int reduce_by_key(long p_key_in, long p_value_in, long p_key_out, long p_value_out);
	public static native int reduce_by_key(long p_key_in, long p_value_in, long p_key_out, long p_value_out, long p_binary_pred);
	public static native int reduce_by_key(long p_key_in, long p_value_in, long p_key_out, long p_value_out, long p_binary_pred, long p_binary_op);
	public static native Boolean equal(long p_vec1, long p_vec2);
	public static native Boolean equal(long p_vec1, long p_vec2, long p_binary_pred);
	public static native int min_element(long p_vec);
	public static native int min_element(long p_vec, long p_comp);
	public static native int max_element(long p_vec);
	public static native int max_element(long p_vec, long p_comp);
	public static native int[] minmax_element(long p_vec);
	public static native int[] minmax_element(long p_vec, long p_comp);
	public static native Object inner_product(long p_vec1, long p_vec2, long p_init);
	public static native Object inner_product(long p_vec1, long p_vec2, long p_init, long p_binary_op1, long p_binary_op2);
	public static native Object transform_reduce(long p_vec, long p_unary_op, long p_init, long p_binary_op);
	public static native Boolean all_of(long p_vec, long p_pred);
	public static native Boolean any_of(long p_vec, long p_pred);
	public static native Boolean none_of(long p_vec, long p_pred);
	public static native Boolean is_partitioned(long p_vec, long p_pred);
	public static native Boolean is_sorted(long p_vec);
	public static native Boolean is_sorted(long p_vec, long p_comp);

	// PrefixSums
	public static native boolean inclusive_scan(long p_vec_in, long p_vec_out);
	public static native boolean inclusive_scan(long p_vec_in, long p_vec_out, long p_binary_op);
	public static native boolean exclusive_scan(long p_vec_in, long p_vec_out);
	public static native boolean exclusive_scan(long p_vec_in, long p_vec_out, long p_init);
	public static native boolean exclusive_scan(long p_vec_in, long p_vec_out, long p_init, long p_binary_op);
	public static native boolean inclusive_scan_by_key(long p_vec_key, long p_vec_value, long p_vec_out);
	public static native boolean inclusive_scan_by_key(long p_vec_key, long p_vec_value, long p_vec_out, long p_binary_pred);
	public static native boolean inclusive_scan_by_key(long p_vec_key, long p_vec_value, long p_vec_out, long p_binary_pred, long p_binary_op);
	public static native boolean exclusive_scan_by_key(long p_vec_key, long p_vec_value, long p_vec_out);
	public static native boolean exclusive_scan_by_key(long p_vec_key, long p_vec_value, long p_vec_out, long p_init);
	public static native boolean exclusive_scan_by_key(long p_vec_key, long p_vec_value, long p_vec_out, long p_init, long p_binary_pred);
	public static native boolean exclusive_scan_by_key(long p_vec_key, long p_vec_value, long p_vec_out, long p_init, long p_binary_pred, long p_binary_op);
	public static native boolean transform_inclusive_scan(long p_vec_in, long p_vec_out, long p_unary_op, long p_binary_op);
	public static native boolean transform_exclusive_scan(long p_vec_in, long p_vec_out, long p_unary_op, long p_init, long p_binary_op);

	// Reordering
	public static native int copy_if(long p_vec_in, long p_vec_out, long p_pred);
	public static native int copy_if_stencil(long p_vec_in, long p_vec_stencil, long p_vec_out, long p_pred);
	public static native int remove(long p_vec, long p_value);
	public static native int remove_copy(long p_vec_in, long p_vec_out, long p_value);
	public static native int remove_if(long p_vec, long p_pred);
	public static native int remove_copy_if(long p_vec_in, long p_vec_out, long p_pred);
	public static native int remove_if_stencil(long p_vec, long p_stencil, long p_pred);
	public static native int remove_copy_if_stencil(long p_vec_in, long p_stencil, long p_vec_out, long p_pred);
	public static native int unique(long p_vec);
	public static native int unique(long p_vec, long p_binary_pred);
	public static native int unique_copy(long p_vec_in, long p_vec_out);
	public static native int unique_copy(long p_vec_in, long p_vec_out, long p_binary_pred);
	public static native int unique_by_key(long p_keys, long p_values);
	public static native int unique_by_key(long p_keys, long p_values, long p_binary_pred);
	public static native int unique_by_key_copy(long p_keys_in, long p_values_in, long p_keys_out, long p_values_out);
	public static native int unique_by_key_copy(long p_keys_in, long p_values_in, long p_keys_out, long p_values_out, long p_binary_pred);
	public static native int partition(long p_vec, long p_pred);
	public static native int partition_stencil(long p_vec, long p_stencil, long p_pred);
	public static native int partition_copy(long p_vec_in, long p_vec_true, long p_vec_false, long p_pred);
	public static native int partition_copy_stencil(long p_vec_in, long p_stencil, long p_vec_true, long p_vec_false, long p_pred);

	// Searching
	public static native Integer find(long p_vec, long p_value);
	public static native Integer find_if(long p_vec, long p_pred);
	public static native Integer find_if_not(long p_vec, long p_pred);
	public static native Integer mismatch(long p_vec1, long p_vec2);
	public static native Integer mismatch(long p_vec1, long p_vec2, long p_pred);
	public static native Integer lower_bound(long p_vec, long p_value);
	public static native Integer lower_bound(long p_vec, long p_value, long p_comp);
	public static native Integer upper_bound(long p_vec, long p_value);
	public static native Integer upper_bound(long p_vec, long p_value, long p_comp);
	public static native Boolean binary_search(long p_vec, long p_value);
	public static native Boolean binary_search(long p_vec, long p_value, long p_comp);
	public static native boolean lower_bound_v(long p_vec, long p_values, long p_result);
	public static native boolean lower_bound_v(long p_vec, long p_values, long p_result, long p_comp);
	public static native boolean upper_bound_v(long p_vec, long p_values, long p_result);
	public static native boolean upper_bound_v(long p_vec, long p_values, long p_result, long p_comp);
	public static native boolean binary_search_v(long p_vec, long p_values, long p_result);
	public static native boolean binary_search_v(long p_vec, long p_values, long p_result, long p_comp);
	public static native Integer partition_point(long p_vec, long p_pred);
	public static native Integer is_sorted_until(long p_vec);
	public static native Integer is_sorted_until(long p_vec, long p_comp);

	// Merging
	public static native boolean merge(long p_vec1, long p_vec2, long p_vec_out);
	public static native boolean merge(long p_vec1, long p_vec2, long p_vec_out, long p_comp);
	public static native boolean merge_by_key(long p_key1, long p_keys2, long p_value1, long p_value2, long p_keys_out, long p_value_out);
	public static native boolean merge_by_key(long p_key1, long p_keys2, long p_value1, long p_value2, long p_keys_out, long p_value_out, long p_comp);

	// Sorting
	public static native boolean sort(long p_vec);
	public static native boolean sort(long p_vec, long p_comp);
	public static native boolean sort_by_key(long p_keys, long p_values);
	public static native boolean sort_by_key(long p_keys, long p_values, long p_comp);
}


