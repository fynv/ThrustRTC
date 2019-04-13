#include <Python.h>
#include <cuda_runtime.h>
#include "TRTCContext.h"

static PyObject* n_set_ptx_cache(PyObject* self, PyObject* args)
{
	const char* path = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	TRTCContext::set_ptx_cache(path);
	return PyLong_FromLong(0);
}

#include "Context.hpp"
#include "DeviceViewable.hpp"
#include "DVVector.hpp"
#include "fill.hpp"
#include "replace.hpp"
#include "for_each.hpp"
#include "adjacent_difference.hpp"
#include "sequence.hpp"

static PyMethodDef s_Methods[] = {
	{ "n_set_ptx_cache", n_set_ptx_cache, METH_VARARGS, "" },
	{ "n_context_create", n_context_create, METH_VARARGS, "" },
	{ "n_context_destroy", n_context_destroy, METH_VARARGS, "" },
	{ "n_context_set_verbose", n_context_set_verbose, METH_VARARGS, "" },
	{ "n_context_add_include_dir", n_context_add_include_dir, METH_VARARGS, "" },
	{ "n_context_add_built_in_header", n_context_add_built_in_header, METH_VARARGS, "" },
	{ "n_context_add_inlcude_filename", n_context_add_inlcude_filename, METH_VARARGS, "" },
	{ "n_context_add_preprocessor", n_context_add_preprocessor, METH_VARARGS, "" },
	{ "n_context_add_constant_object", n_context_add_constant_object, METH_VARARGS, "" },
	{ "n_context_launch_kernel", n_context_launch_kernel, METH_VARARGS, "" },
	{ "n_context_launch_for", n_context_launch_for, METH_VARARGS, "" },
	{ "n_kernel_create", n_kernel_create, METH_VARARGS, "" },
	{ "n_kernel_destroy", n_kernel_destroy, METH_VARARGS, "" },
	{ "n_kernel_num_params", n_kernel_num_params, METH_VARARGS, "" },
	{ "n_kernel_launch", n_kernel_launch, METH_VARARGS, "" },
	{ "n_for_create", n_for_create, METH_VARARGS, "" },
	{ "n_for_destroy", n_for_destroy, METH_VARARGS, "" },
	{ "n_for_num_params", n_for_num_params, METH_VARARGS, "" },
	{ "n_for_launch", n_for_launch, METH_VARARGS, "" },
	{ "n_dv_name_view_cls", n_dv_name_view_cls, METH_VARARGS, "" },
	{ "n_dv_destroy", n_dv_destroy, METH_VARARGS, "" },
	{ "n_dv_create_basic", n_dv_create_basic, METH_VARARGS, "" },
	{ "n_dvvector_create", n_dvvector_create, METH_VARARGS, "" },
	{ "n_dvvector_name_elem_cls", n_dvvector_name_elem_cls, METH_VARARGS, "" },
	{ "n_dvvector_size", n_dvvector_size, METH_VARARGS, "" },
	{ "n_dvvector_to_host", n_dvvector_to_host, METH_VARARGS, "" },
	{ "n_dvvector_from_dvs", n_dvvector_from_dvs, METH_VARARGS, "" },
	{ "n_functor_generate_code", n_functor_generate_code, METH_VARARGS, "" },
	{ "n_fill", n_fill, METH_VARARGS, "" },
	{ "n_replace", n_replace, METH_VARARGS, "" },
	{ "n_replace_if", n_replace_if, METH_VARARGS, "" },
	{ "n_replace_copy", n_replace_copy, METH_VARARGS, "" },
	{ "n_replace_copy_if", n_replace_copy_if, METH_VARARGS, "" },
	{ "n_for_each", n_for_each, METH_VARARGS, "" },
	{ "n_adjacent_difference", n_adjacent_difference, METH_VARARGS, "" },
	{ "n_sequence", n_sequence, METH_VARARGS, "" },
	0
};

static struct PyModuleDef cModPyDem = { PyModuleDef_HEAD_INIT, "ThrustRTC_module", "", -1, s_Methods };

PyMODINIT_FUNC PyInit_PyThrustRTC(void)
{
	cudaFree(0);
	return PyModule_Create(&cModPyDem);
}
