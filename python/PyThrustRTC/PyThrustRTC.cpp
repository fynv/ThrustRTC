#include <Python.h>
#include "TRTCContext.h"

static PyObject* n_set_libnvrtc_path(PyObject* self, PyObject* args)
{
	const char* path = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	TRTCContext::set_libnvrtc_path(path);
	return PyLong_FromLong(0);
}

#include "Context.hpp"
#include "DeviceViewable.hpp"
#include "DVVector.hpp"
#include "DVTuple.hpp"
#include "DVConstant.hpp"
#include "DVCounter.hpp"
#include "DVDiscard.hpp"
#include "DVPermutation.hpp"
#include "DVReverse.hpp"
#include "DVTransform.hpp"
#include "DVZipped.hpp"
#include "functor.hpp"
#include "fill.hpp"
#include "replace.hpp"
#include "for_each.hpp"
#include "adjacent_difference.hpp"
#include "sequence.hpp"
#include "tabulate.hpp"
#include "transform.hpp"
#include "gather.hpp"
#include "scatter.hpp"
#include "copy.hpp"
#include "swap.hpp"
#include "count.hpp"
#include "reduce.hpp"
#include "equal.hpp"
#include "extrema.hpp"
#include "inner_product.hpp"
#include "transform_reduce.hpp"
#include "logical.hpp"
#include "scan.hpp"
#include "transform_scan.hpp"
#include "remove.hpp"
#include "unique.hpp"
#include "partition.hpp"
#include "find.hpp"
#include "mismatch.hpp"
#include "binary_search.hpp"

static PyMethodDef s_Methods[] = {
	{ "n_set_libnvrtc_path", n_set_libnvrtc_path, METH_VARARGS, "" },
	{ "n_context_create", n_context_create, METH_VARARGS, "" },
	{ "n_context_destroy", n_context_destroy, METH_VARARGS, "" },
	{ "n_context_set_verbose", n_context_set_verbose, METH_VARARGS, "" },
	{ "n_context_add_include_dir", n_context_add_include_dir, METH_VARARGS, "" },
	{ "n_context_add_built_in_header", n_context_add_built_in_header, METH_VARARGS, "" },
	{ "n_context_add_inlcude_filename", n_context_add_inlcude_filename, METH_VARARGS, "" },
	{ "n_context_add_code_block", n_context_add_code_block, METH_VARARGS, "" },
	{ "n_context_add_constant_object", n_context_add_constant_object, METH_VARARGS, "" },
	{ "n_context_calc_optimal_block_size", n_context_calc_optimal_block_size, METH_VARARGS, "" },
	{ "n_context_calc_number_blocks", n_context_calc_number_blocks, METH_VARARGS, "" },
	{ "n_context_launch_kernel", n_context_launch_kernel, METH_VARARGS, "" },
	{ "n_context_launch_for", n_context_launch_for, METH_VARARGS, "" },
	{ "n_context_launch_for_n", n_context_launch_for_n, METH_VARARGS, "" },
	{ "n_kernel_create", n_kernel_create, METH_VARARGS, "" },
	{ "n_kernel_destroy", n_kernel_destroy, METH_VARARGS, "" },
	{ "n_kernel_num_params", n_kernel_num_params, METH_VARARGS, "" },
	{ "n_kernel_calc_optimal_block_size", n_kernel_calc_optimal_block_size, METH_VARARGS, "" },
	{ "n_kernel_calc_number_blocks", n_kernel_calc_number_blocks, METH_VARARGS, "" },
	{ "n_kernel_launch", n_kernel_launch, METH_VARARGS, "" },
	{ "n_for_create", n_for_create, METH_VARARGS, "" },
	{ "n_for_destroy", n_for_destroy, METH_VARARGS, "" },
	{ "n_for_num_params", n_for_num_params, METH_VARARGS, "" },
	{ "n_for_launch", n_for_launch, METH_VARARGS, "" },
	{ "n_for_launch_n", n_for_launch_n, METH_VARARGS, "" },
	{ "n_dv_name_view_cls", n_dv_name_view_cls, METH_VARARGS, "" },
	{ "n_dv_destroy", n_dv_destroy, METH_VARARGS, "" },
	{ "n_dv_create_basic", n_dv_create_basic, METH_VARARGS, "" },
	{ "n_dv_value", n_dv_value, METH_VARARGS, "" },
	{ "n_dvvectorlike_name_elem_cls", n_dvvectorlike_name_elem_cls, METH_VARARGS, "" },
	{ "n_dvvectorlike_size", n_dvvectorlike_size, METH_VARARGS, "" },
	{ "n_dvvector_create", n_dvvector_create, METH_VARARGS, "" },
	{ "n_dvvector_to_host", n_dvvector_to_host, METH_VARARGS, "" },
	{ "n_dvvector_from_dvs", n_dvvector_from_dvs, METH_VARARGS, "" },
	{ "n_dvtuple_create", n_dvtuple_create, METH_VARARGS, "" },
	{ "n_dvconstant_create", n_dvconstant_create, METH_VARARGS, "" },
	{ "n_dvcounter_create", n_dvcounter_create, METH_VARARGS, "" },
	{ "n_dvdiscard_create", n_dvdiscard_create, METH_VARARGS, "" },
	{ "n_dvpermutation_create", n_dvpermutation_create, METH_VARARGS, "" },
	{ "n_dvreverse_create", n_dvreverse_create, METH_VARARGS, "" },
	{ "n_dvtransform_create", n_dvtransform_create, METH_VARARGS, "" },
	{ "n_dvzipped_create", n_dvzipped_create, METH_VARARGS, "" },
	{ "n_functor_create", n_functor_create, METH_VARARGS, "" },
	{ "n_built_in_functor_create", n_built_in_functor_create, METH_VARARGS, "" },
	{ "n_fill", n_fill, METH_VARARGS, "" },
	{ "n_replace", n_replace, METH_VARARGS, "" },
	{ "n_replace_if", n_replace_if, METH_VARARGS, "" },
	{ "n_replace_copy", n_replace_copy, METH_VARARGS, "" },
	{ "n_replace_copy_if", n_replace_copy_if, METH_VARARGS, "" },
	{ "n_for_each", n_for_each, METH_VARARGS, "" },
	{ "n_adjacent_difference", n_adjacent_difference, METH_VARARGS, "" },
	{ "n_sequence", n_sequence, METH_VARARGS, "" },
	{ "n_tabulate", n_tabulate, METH_VARARGS, "" },
	{ "n_transform", n_transform, METH_VARARGS, "" },
	{ "n_transform_binary", n_transform_binary, METH_VARARGS, "" },
	{ "n_transform_if", n_transform_if, METH_VARARGS, "" },
	{ "n_transform_if_stencil", n_transform_if_stencil, METH_VARARGS, "" },
	{ "n_transform_binary_if_stencil", n_transform_binary_if_stencil, METH_VARARGS, "" },
	{ "n_gather", n_gather, METH_VARARGS, "" },
	{ "n_gather_if", n_gather_if, METH_VARARGS, "" },
	{ "n_scatter", n_scatter, METH_VARARGS, "" },
	{ "n_scatter_if", n_scatter_if, METH_VARARGS, "" },
	{ "n_copy", n_copy, METH_VARARGS, "" },
	{ "n_swap", n_swap, METH_VARARGS, "" },
	{ "n_count", n_count, METH_VARARGS, "" },
	{ "n_count_if", n_count_if, METH_VARARGS, "" },
	{ "n_reduce", n_reduce, METH_VARARGS, "" },
	{ "n_equal", n_equal, METH_VARARGS, "" },
	{ "n_min_element", n_min_element, METH_VARARGS, "" },
	{ "n_max_element", n_max_element, METH_VARARGS, "" },
	{ "n_minmax_element", n_minmax_element, METH_VARARGS, "" },
	{ "n_inner_product", n_inner_product, METH_VARARGS, "" },
	{ "n_transform_reduce", n_transform_reduce, METH_VARARGS, "" },
	{ "n_all_of", n_all_of, METH_VARARGS, "" },
	{ "n_any_of", n_any_of, METH_VARARGS, "" },
	{ "n_none_of", n_none_of, METH_VARARGS, "" },
	{ "n_inclusive_scan", n_inclusive_scan, METH_VARARGS, "" },
	{ "n_exclusive_scan", n_exclusive_scan, METH_VARARGS, "" },
	{ "n_transform_inclusive_scan", n_transform_inclusive_scan, METH_VARARGS, "" },
	{ "n_transform_exclusive_scan", n_transform_exclusive_scan, METH_VARARGS, "" },
	{ "n_inclusive_scan_by_key", n_inclusive_scan_by_key, METH_VARARGS, "" },
	{ "n_exclusive_scan_by_key", n_exclusive_scan_by_key, METH_VARARGS, "" },
	{ "n_copy_if", n_copy_if, METH_VARARGS, "" },
	{ "n_copy_if_stencil", n_copy_if_stencil, METH_VARARGS, "" },
	{ "n_remove", n_remove, METH_VARARGS, "" },
	{ "n_remove_copy", n_remove_copy, METH_VARARGS, "" },
	{ "n_remove_if", n_remove_if, METH_VARARGS, "" },
	{ "n_remove_copy_if", n_remove_copy_if, METH_VARARGS, "" },
	{ "n_remove_if_stencil", n_remove_if_stencil, METH_VARARGS, "" },
	{ "n_remove_copy_if_stencil", n_remove_copy_if_stencil, METH_VARARGS, "" },
	{ "n_unique", n_unique, METH_VARARGS, "" },
	{ "n_unique_copy", n_unique_copy, METH_VARARGS, "" },
	{ "n_unique_by_key", n_unique_by_key, METH_VARARGS, "" },
	{ "n_unique_by_key_copy", n_unique_by_key_copy, METH_VARARGS, "" },
	{ "n_partition", n_partition, METH_VARARGS, "" },
	{ "n_partition_stencil", n_partition_stencil, METH_VARARGS, "" },
	{ "n_partition_copy", n_partition_copy, METH_VARARGS, "" },
	{ "n_partition_copy_stencil", n_partition_copy_stencil, METH_VARARGS, "" },
	{ "n_find", n_find, METH_VARARGS, "" },
	{ "n_find_if", n_find_if, METH_VARARGS, "" },
	{ "n_find_if_not", n_find_if_not, METH_VARARGS, "" },
	{ "n_mismatch", n_mismatch, METH_VARARGS, "" },
	{ "n_lower_bound", n_lower_bound, METH_VARARGS, "" },
	{ "n_upper_bound", n_upper_bound, METH_VARARGS, "" },
	{ "n_binary_search", n_binary_search, METH_VARARGS, "" },
	0
};

static struct PyModuleDef cModPyDem = { PyModuleDef_HEAD_INIT, "ThrustRTC_module", "", -1, s_Methods };

PyMODINIT_FUNC PyInit_PyThrustRTC(void)
{
	return PyModule_Create(&cModPyDem);
}
