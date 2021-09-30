import os
if os.path.exists('ThrustRTC/cffi.py'):
	os.remove('ThrustRTC/cffi.py')
	
import cffi

ffibuilder = cffi.FFI()
ffibuilder.set_source("ThrustRTC.cffi", None)

ffibuilder.cdef("""    
// utils
void* n_string_array_create(unsigned long long size, const char* const* strs);
void n_string_array_destroy(void* ptr_arr);
void* n_pointer_array_create(unsigned long long size, const void* const* ptrs);
void n_pointer_array_destroy(void* ptr_arr);
void* n_dim3_create(unsigned x, unsigned y, unsigned z);
void n_dim3_destroy(void* cptr);

// Context
void n_set_libnvrtc_path(const char* path);
int n_trtc_try_init();
int n_get_ptx_arch();
void n_set_verbose(unsigned verbose);
void n_add_include_dir(const char* dir);
void n_add_built_in_header(const char* filename, const char* filecontent);
void n_add_inlcude_filename(const char* fn);
void n_add_code_block(const char* line);
void n_add_constant_object(const char* name, void* cptr);
void n_wait();

void* n_kernel_create(void* ptr_param_list, const char* body);
void n_kernel_destroy(void* cptr);
int n_kernel_num_params(void* cptr);
int n_kernel_calc_optimal_block_size(void* ptr_kernel, void* ptr_arg_list, unsigned sharedMemBytes);
int n_kernel_calc_number_blocks(void* ptr_kernel, void* ptr_arg_list, int sizeBlock, unsigned sharedMemBytes);
int n_kernel_launch(void* ptr_kernel, void* ptr_gridDim, void* ptr_blockDim, void* ptr_arg_list, int sharedMemBytes);

void* n_for_create(void* ptr_param_list, const char* name_iter, const char* body);
void n_for_destroy(void* cptr);
int n_for_num_params(void* cptr);
int n_for_launch(void* cptr, int begin, int end, void* ptr_arg_list);
int n_for_launch_n(void* cptr, int n, void* ptr_arg_list);

// DeviceViewable
const char* n_dv_name_view_cls(void* cptr);
void n_dv_destroy(void* cptr);
void* n_dvint8_create(int v);
int n_dvint8_value(void* cptr);
void* n_dvuint8_create(unsigned v);
unsigned n_dvuint8_value(void* cptr);
void* n_dvint16_create(int v);
int n_dvint16_value(void* cptr);
void* n_dvuint16_create(unsigned v);
unsigned n_dvuint16_value(void* cptr);
void* n_dvint32_create(int v);
int n_dvint32_value(void* cptr);
void* n_dvuint32_create(unsigned v);
unsigned n_dvuint32_value(void* cptr);
void* n_dvint64_create(long long v);
long long n_dvint64_value(void* cptr);
void* n_dvuint64_create(unsigned long long v);
unsigned long long n_dvuint64_value(void* cptr);
void* n_dvfloat_create(float v);
float n_dvfloat_value(void* cptr);
void* n_dvdouble_create(double v);
double n_dvdouble_value(void* cptr);
void* n_dvbool_create(int v);
int n_dvbool_value(void* cptr);
void* n_dvcomplex64_create(float real, float imag);
float n_dvcomplex64_real(void* cptr);
float n_dvcomplex64_imag(void* cptr);
void* n_dvcomplex128_create(double real, double imag);
double n_dvcomplex128_real(void* cptr);
double n_dvcomplex128_imag(void* cptr);

// DVVector
const char* n_dvvectorlike_name_elem_cls(void* cptr);
unsigned long long n_dvvectorlike_size(void* cptr);
void* n_dvvector_create(const char* elem_cls, unsigned long long size, void* hdata);
void n_dvvector_to_host(void* cptr, void* hdata, unsigned long long begin, unsigned long long end);
void* n_dvvector_from_dvs(void* ptr_dvs);
void* n_dvvectoradaptor_create(const char* elem_cls, unsigned long long size, void* data);
void* n_dvrange_create(void* ptr_in, unsigned long long begin, unsigned long long end);

// Tuple
void* n_dvtuple_create(void* ptr_dvs, void* ptr_names);

// Fake-Vectors
void* n_dvconstant_create(void* cptr, unsigned long long size);
void* n_dvcounter_create(void* cptr, unsigned long long size);
void* n_dvdiscard_create(const char* elem_cls, unsigned long long size);
void* n_dvpermutation_create(void* ptr_value, void* ptr_index);
void* n_dvreverse_create(void* ptr_value);
void* n_dvtransform_create(void* ptr_in, const char* elem_cls, void* ptr_op);
void* n_dvzipped_create(void* ptr_vecs, void* ptr_elem_names);
void* n_dvcustomvector_create(void* ptr_dvs, void* ptr_names, const char* name_idx, const char* body, const char* elem_cls, unsigned long long size, unsigned read_only);

// Functor
void* n_functor_create(void* ptr_dvs, void* ptr_names, void* ptr_functor_params, const char* code_body);
void* n_built_in_functor_create(const char* name_built_in_view_cls);

// Transformations
int n_fill(void* ptr_vec, void* ptr_value);
int n_replace(void* ptr_vec, void* ptr_old_value, void* ptr_new_value);
int n_replace_if(void* ptr_vec, void* p_pred, void* ptr_new_value);
int n_replace_copy(void* ptr_in, void* ptr_out, void* ptr_old_value, void* ptr_new_value);
int n_replace_copy_if(void* ptr_in, void* ptr_out, void* p_pred, void* ptr_new_value);
int n_for_each(void* ptr_vec, void* ptr_f);
int n_adjacent_difference(void* ptr_in, void* ptr_out, void* ptr_binary_op);
int n_sequence(void* ptr_vec, void* ptr_value_init, void* ptr_value_step);
int n_tabulate(void* ptr_vec, void* ptr_op);
int n_transform(void* ptr_in, void* ptr_out, void* ptr_op);
int n_transform_binary(void* ptr_in1, void* ptr_in2, void* ptr_out, void* ptr_op);
int n_transform_if(void* ptr_in, void* ptr_out, void* ptr_op, void* ptr_pred);
int n_transform_if_stencil(void* ptr_in, void* ptr_stencil, void* ptr_out, void* ptr_op, void* ptr_pred);
int n_transform_binary_if_stencil(void* ptr_in1, void* ptr_in2, void* ptr_stencil, void* ptr_out, void* ptr_op, void* ptr_pred);

// Copying
int n_gather(void* ptr_map, void* ptr_in, void* ptr_out);
int n_gather_if(void* ptr_map, void* ptr_stencil, void* ptr_in, void* ptr_out, void* ptr_pred);
int n_scatter(void* ptr_in, void* ptr_map, void* ptr_out);
int n_scatter_if(void* ptr_in, void* ptr_map, void* ptr_stencil, void* ptr_out, void* ptr_pred);
int n_copy(void* ptr_in, void* ptr_out);
int n_swap(void* ptr_vec1, void* ptr_vec2);

// Redutions
unsigned long long n_count(void* ptr_vec, void* ptr_value);
unsigned long long n_count_if(void* ptr_vec, void* ptr_pred);
void* n_reduce(void* ptr_vec, void* ptr_init, void* ptr_bin_op);
unsigned n_reduce_by_key(void* ptr_key_in, void* ptr_value_in, void* ptr_key_out, void* ptr_value_out, void* ptr_binary_pred, void* ptr_binary_op);
int n_equal(void* ptr_vec1, void* ptr_vec2, void* ptr_binary_pred);
unsigned long long n_min_element(void* ptr_vec, void* ptr_comp);
unsigned long long n_max_element(void* ptr_vec, void* ptr_comp);
int n_minmax_element(void* ptr_vec, void* ptr_comp, unsigned long long* ret);
void* n_inner_product(void* ptr_vec1, void* ptr_vec2, void* ptr_init, void* ptr_binary_op1, void* ptr_binary_op2);
void* n_transform_reduce(void* ptr_vec, void* ptr_unary_op, void* ptr_init, void* ptr_binary_op);
int n_all_of(void* ptr_vec, void* ptr_pred);
int n_any_of(void* ptr_vec, void* ptr_pred);
int n_none_of(void* ptr_vec, void* ptr_pred);
int n_is_partitioned(void* ptr_vec, void* ptr_pred);
int n_is_sorted(void* ptr_vec, void* ptr_comp);

// PrefixSums
int n_inclusive_scan(void* ptr_vec_in, void* ptr_vec_out, void* ptr_binary_op);
int n_exclusive_scan(void* ptr_vec_in, void* ptr_vec_out, void* ptr_init, void* ptr_binary_op);
int n_inclusive_scan_by_key(void* ptr_vec_key, void* ptr_vec_value, void* ptr_vec_out, void* ptr_binary_pred, void* ptr_binary_op);
int n_exclusive_scan_by_key(void* ptr_vec_key, void* ptr_vec_value, void* ptr_vec_out, void* ptr_init, void* ptr_binary_pred, void* ptr_binary_op);
int n_transform_inclusive_scan(void* ptr_vec_in, void* ptr_vec_out, void* ptr_unary_op, void* ptr_binary_op);
int n_transform_exclusive_scan(void* ptr_vec_in, void* ptr_vec_out, void* ptr_unary_op, void* ptr_init, void* ptr_binary_op);

// Reordering
unsigned n_copy_if(void* ptr_in, void* ptr_out, void* ptr_pred);
unsigned n_copy_if_stencil(void* ptr_in, void* ptr_stencil, void* ptr_out, void* ptr_pred);
unsigned n_remove(void* ptr_vec, void* ptr_value);
unsigned n_remove_copy(void* ptr_in, void* ptr_out, void* ptr_value);
unsigned n_remove_if(void* ptr_vec, void* ptr_pred);
unsigned n_remove_copy_if(void* ptr_in, void* ptr_out, void* ptr_pred);
unsigned n_remove_if_stencil(void* ptr_vec, void* ptr_stencil, void* ptr_pred);
unsigned n_remove_copy_if_stencil(void* ptr_in, void* ptr_stencil, void* ptr_out, void* ptr_pred);
unsigned n_unique(void* ptr_vec, void* ptr_binary_pred);
unsigned n_unique_copy(void* ptr_vec_in, void* ptr_vec_out, void* ptr_binary_pred);
unsigned n_unique_by_key(void* ptr_keys, void* ptr_values, void* ptr_binary_pred);
unsigned n_unique_by_key_copy(void* ptr_keys_in, void* ptr_values_in, void* ptr_key_out, void* ptr_values_out, void* ptr_binary_pred);
unsigned n_partition(void* ptr_vec, void* ptr_pred);
unsigned n_partition_stencil(void* ptr_vec, void* ptr_stencil, void* ptr_pred);
unsigned n_partition_copy(void* ptr_vec_in, void* ptr_vec_true, void* ptr_vec_false, void* ptr_pred);
unsigned n_partition_copy_stencil(void* ptr_vec_in, void* ptr_stencil, void* ptr_vec_true, void* ptr_vec_false, void* ptr_pred);

// Searching
long long n_find(void* ptr_vec, void* ptr_value);
long long n_find_if(void* ptr_vec, void* ptr_pred);
long long n_find_if_not(void* ptr_vec, void* ptr_pred);
long long n_mismatch(void* ptr_vec1, void* ptr_vec2, void* ptr_pred);
unsigned long long n_lower_bound(void* ptr_vec, void* ptr_value, void* ptr_comp);
unsigned long long n_upper_bound(void* ptr_vec, void* ptr_value, void* ptr_comp);
int n_binary_search(void* ptr_vec, void* ptr_value, void* ptr_comp);
int n_lower_bound_v(void* ptr_vec, void* ptr_values, void* ptr_result, void* ptr_comp);
int n_upper_bound_v(void* ptr_vec, void* ptr_values, void* ptr_result, void* ptr_comp);
int n_binary_search_v(void* ptr_vec, void* ptr_values, void* ptr_result, void* ptr_comp);
unsigned long long n_partition_point(void* ptr_vec, void* ptr_pred);
unsigned long long n_is_sorted_until(void* ptr_vec, void* ptr_comp);

// Merging	
int n_merge(void* ptr_vec1, void* ptr_vec2, void* ptr_vec_out, void* ptr_comp);
int n_merge_by_key(void* ptr_keys1, void* ptr_keys2, void* ptr_value1, void* ptr_value2, void* ptr_keys_out, void* ptr_value_out, void* ptr_comp);

// Sorting
int n_sort(void* ptr_vec, void* ptr_comp);
int n_sort_by_key(void* ptr_keys, void* ptr_values, void* ptr_comp);
""")


ffibuilder.compile()
