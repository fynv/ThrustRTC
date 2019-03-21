#include <Python.h>
#include "TRTCContext.h"
#include "CachedKernelTemplate.h"

static PyObject* n_context_create(PyObject* self, PyObject* args)
{
	TRTCContext *ctx = new TRTCContext;
	return PyLong_FromVoidPtr(ctx);
}

static PyObject* n_context_destroy(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete ctx;
	return PyLong_FromLong(0);
}

static PyObject* n_context_set_verbose(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	bool verbose = PyObject_IsTrue(PyTuple_GetItem(args, 1)) != 0;
	ctx->set_verbose(verbose);

	return PyLong_FromLong(0);
}


static PyObject* n_context_add_include_dir(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* dir = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	ctx->add_include_dir(dir);

	return PyLong_FromLong(0);
}

static PyObject* n_context_add_inlcude_filename(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* fn = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	ctx->add_inlcude_filename(fn);

	return PyLong_FromLong(0);
}

static PyObject* n_context_add_preprocessor(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* line = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	ctx->add_preprocessor(line);
	return PyLong_FromLong(0);
}

static PyObject* n_context_launch_once(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	dim_type gridDim;
	PyObject* arg1 = PyTuple_GetItem(args, 1);
	if (PyObject_TypeCheck(arg1, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg1);
		gridDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 0)) : 1;
		gridDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 1)) : 1;
		gridDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 2)) : 1;
	}
	else
	{
		gridDim.x = (unsigned)PyLong_AsUnsignedLong(arg1);
		gridDim.y = 1;
		gridDim.z = 1;
	}

	dim_type blockDim;
	PyObject* arg2 = PyTuple_GetItem(args, 2);
	if (PyObject_TypeCheck(arg2, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg2);
		blockDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 0)) : 1;
		blockDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 1)) : 1;
		blockDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 2)) : 1;
	}
	else
	{
		blockDim.x = (unsigned)PyLong_AsUnsignedLong(arg2);
		blockDim.y = 1;
		blockDim.z = 1;
	}

	PyObject* pyArgMap = PyTuple_GetItem(args, 3);
	ssize_t num_params = PyList_Size(pyArgMap);
	std::vector<TRTCContext::AssignedParam> arg_map(num_params);
	for (ssize_t i = 0; i < num_params; i++)
	{
		PyObject* pyAssignedParam = PyList_GetItem(pyArgMap, i);
		arg_map[i].param_name = PyUnicode_AsUTF8(PyTuple_GetItem(pyAssignedParam, 0));
		arg_map[i].arg = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(pyAssignedParam, 1));
	}

	const char* code_body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 4));
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 5));
	ctx->launch_once(gridDim, blockDim, arg_map, code_body, sharedMemBytes);

	return PyLong_FromLong(0);
}


static PyObject* n_kernel_create(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* pyParamList = PyTuple_GetItem(args, 1);
	ssize_t num_params = PyList_Size(pyParamList);
	std::vector<TRTCContext::ParamDesc> params(num_params);
	for (ssize_t i = 0; i < num_params; i++)
	{
		PyObject* tuple = PyList_GetItem(pyParamList, i);
		params[i].type = PyUnicode_AsUTF8(PyTuple_GetItem(tuple, 0));
		params[i].name = PyUnicode_AsUTF8(PyTuple_GetItem(tuple, 1));
	}
	const char* body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 2));

	TRTCContext::Kernel* kernel = ctx->create_kernel(params, body);
	if (kernel == nullptr)
	{
		PyErr_Format(PyExc_ValueError, "Failed to compile kernel");
		Py_RETURN_NONE;
	}
	return PyLong_FromVoidPtr((void*)kernel);
}

static PyObject* n_kernel_destroy(PyObject* self, PyObject* args)
{
	TRTCContext::Kernel* kernel = (TRTCContext::Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	TRTCContext::destroy_kernel(kernel);
	return PyLong_FromLong(0);
}

static PyObject* n_kernel_num_params(PyObject* self, PyObject* args)
{
	TRTCContext::Kernel* kernel = (TRTCContext::Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLong((long)TRTCContext::get_num_of_params(kernel));
}

static PyObject* n_kernel_launch(PyObject* self, PyObject* args)
{
	TRTCContext::Kernel* kernel = (TRTCContext::Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t num_params = TRTCContext::get_num_of_params(kernel);

	dim_type gridDim;
	PyObject* arg1 = PyTuple_GetItem(args, 1);
	if (PyObject_TypeCheck(arg1, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg1);
		gridDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 0)) : 1;
		gridDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 1)) : 1;
		gridDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 2)) : 1;
	}
	else
	{
		gridDim.x = (unsigned)PyLong_AsUnsignedLong(arg1);
		gridDim.y = 1;
		gridDim.z = 1;
	}

	dim_type blockDim;
	PyObject* arg2 = PyTuple_GetItem(args, 2);
	if (PyObject_TypeCheck(arg2, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg2);
		blockDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 0)) : 1;
		blockDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 1)) : 1;
		blockDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 2)) : 1;
	}
	else
	{
		blockDim.x = (unsigned)PyLong_AsUnsignedLong(arg2);
		blockDim.y = 1;
		blockDim.z = 1;
	}

	PyObject* arg3 = PyTuple_GetItem(args, 3);
	std::vector<DeviceViewable*> params;
	if (PyObject_TypeCheck(arg3, &PyList_Type))
	{
		ssize_t size = PyList_Size(arg3);
		if (num_params != size)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, size);
			Py_RETURN_NONE;
		}
		params.resize(size);
		for (ssize_t i = 0; i < size; i++)
			params[i] = (DeviceViewable*)PyLong_AsVoidPtr(PyList_GetItem(arg3, i));
	}
	else if (arg3 != Py_None)
	{
		if (num_params != 1)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, 1);
			Py_RETURN_NONE;
		}
		params.resize(1);
		params[0] = (DeviceViewable*)PyLong_AsVoidPtr(arg3);
	}
	else
	{
		if (num_params != 0)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, 0);
			Py_RETURN_NONE;
		}
	}
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 4));
	TRTCContext::launch_kernel(kernel, gridDim, blockDim, params.data(), sharedMemBytes);
	return PyLong_FromLong(0);
}


static PyObject* n_kernel_template_create(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* pyTemplParamList = PyTuple_GetItem(args, 1);
	ssize_t num_templ_params = PyList_Size(pyTemplParamList);
	std::vector<const char*> templ_params(num_templ_params);
	for (ssize_t i = 0; i < num_templ_params; i++)
		templ_params[i] = PyUnicode_AsUTF8(PyList_GetItem(pyTemplParamList, i));

	PyObject* pyParamList = PyTuple_GetItem(args, 2);
	ssize_t num_params = PyList_Size(pyParamList);
	std::vector<TRTCContext::ParamDesc> params(num_params);
	for (ssize_t i = 0; i < num_params; i++)
	{
		PyObject* tuple = PyList_GetItem(pyParamList, i);
		params[i].type = PyUnicode_AsUTF8(PyTuple_GetItem(tuple, 0));
		params[i].name = PyUnicode_AsUTF8(PyTuple_GetItem(tuple, 1));
	}
	const char* body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 3));

	CachedKernelTemplate* ktempl = new CachedKernelTemplate(ctx, templ_params, params, body);
	return PyLong_FromVoidPtr((void*)ktempl);
}

static PyObject* n_kernel_template_destroy(PyObject* self, PyObject* args)
{
	CachedKernelTemplate* kernel = (CachedKernelTemplate*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete kernel;
	return PyLong_FromLong(0);
}

static PyObject* n_kernel_template_num_template_params(PyObject* self, PyObject* args)
{
	CachedKernelTemplate* kernel = (CachedKernelTemplate*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLong((long)kernel->num_template_params());
}


static PyObject* n_kernel_template_num_params(PyObject* self, PyObject* args)
{
	CachedKernelTemplate* kernel = (CachedKernelTemplate*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLong((long)kernel->num_params());
}

static PyObject* n_kernel_template_launch_explict(PyObject* self, PyObject* args)
{
	CachedKernelTemplate* kernel = (CachedKernelTemplate*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t num_template_params = kernel->num_template_params();
	size_t num_params = kernel->num_params();

	dim_type gridDim;
	PyObject* arg1 = PyTuple_GetItem(args, 1);
	if (PyObject_TypeCheck(arg1, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg1);
		gridDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 0)) : 1;
		gridDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 1)) : 1;
		gridDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 2)) : 1;
	}
	else
	{
		gridDim.x = (unsigned)PyLong_AsUnsignedLong(arg1);
		gridDim.y = 1;
		gridDim.z = 1;
	}

	dim_type blockDim;
	PyObject* arg2 = PyTuple_GetItem(args, 2);
	if (PyObject_TypeCheck(arg2, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg2);
		blockDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 0)) : 1;
		blockDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 1)) : 1;
		blockDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 2)) : 1;
	}
	else
	{
		blockDim.x = (unsigned)PyLong_AsUnsignedLong(arg2);
		blockDim.y = 1;
		blockDim.z = 1;
	}

	PyObject* arg3 = PyTuple_GetItem(args, 3);
	std::vector<std::string> template_args;
	if (PyObject_TypeCheck(arg3, &PyList_Type))
	{
		ssize_t size = PyList_Size(arg3);
		if (num_template_params != size)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of template arguments received. %d required, %d received.", num_template_params, size);
			Py_RETURN_NONE;
		}
		template_args.resize(size);
		for (ssize_t i = 0; i < size; i++)
			template_args[i] = PyUnicode_AsUTF8(PyList_GetItem(arg3, i));
	}
	else if (arg3 != Py_None)
	{
		if (num_template_params != 1)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of template arguments received. %d required, %d received.", num_template_params, 1);
			Py_RETURN_NONE;
		}
		template_args.resize(1);
		template_args[0] = PyUnicode_AsUTF8(arg3);
	}
	else
	{
		if (num_template_params != 0)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of template arguments received. %d required, %d received.", num_template_params, 0);
			Py_RETURN_NONE;
		}
	}

	PyObject* arg4 = PyTuple_GetItem(args, 4);
	std::vector<DeviceViewable*> params;
	if (PyObject_TypeCheck(arg4, &PyList_Type))
	{
		ssize_t size = PyList_Size(arg4);
		if (num_params != size)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, size);
			Py_RETURN_NONE;
		}
		params.resize(size);
		for (ssize_t i = 0; i < size; i++)
			params[i] = (DeviceViewable*)PyLong_AsVoidPtr(PyList_GetItem(arg4, i));
	}
	else if (arg4 != Py_None)
	{
		if (num_params != 1)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, 1);
			Py_RETURN_NONE;
		}
		params.resize(1);
		params[0] = (DeviceViewable*)PyLong_AsVoidPtr(arg4);
	}
	else
	{
		if (num_params != 0)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, 0);
			Py_RETURN_NONE;
		}
	}
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 5));
	kernel->launch(gridDim, blockDim, template_args, params.data(), sharedMemBytes);
	return PyLong_FromLong(0);
}

static PyObject* n_kernel_template_launch(PyObject* self, PyObject* args)
{
	CachedKernelTemplate* kernel = (CachedKernelTemplate*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t num_params = kernel->num_params();
	dim_type gridDim;
	PyObject* arg1 = PyTuple_GetItem(args, 1);
	if (PyObject_TypeCheck(arg1, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg1);
		gridDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 0)) : 1;
		gridDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 1)) : 1;
		gridDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 2)) : 1;
	}
	else
	{
		gridDim.x = (unsigned)PyLong_AsUnsignedLong(arg1);
		gridDim.y = 1;
		gridDim.z = 1;
	}

	dim_type blockDim;
	PyObject* arg2 = PyTuple_GetItem(args, 2);
	if (PyObject_TypeCheck(arg2, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg2);
		blockDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 0)) : 1;
		blockDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 1)) : 1;
		blockDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 2)) : 1;
	}
	else
	{
		blockDim.x = (unsigned)PyLong_AsUnsignedLong(arg2);
		blockDim.y = 1;
		blockDim.z = 1;
	}

	PyObject* arg3 = PyTuple_GetItem(args, 3);
	std::vector<DeviceViewable*> params;
	if (PyObject_TypeCheck(arg3, &PyList_Type))
	{
		ssize_t size = PyList_Size(arg3);
		if (num_params != size)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, size);
			Py_RETURN_NONE;
		}
		params.resize(size);
		for (ssize_t i = 0; i < size; i++)
			params[i] = (DeviceViewable*)PyLong_AsVoidPtr(PyList_GetItem(arg3, i));
	}
	else if (arg3 != Py_None)
	{
		if (num_params != 1)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, 1);
			Py_RETURN_NONE;
		}
		params.resize(1);
		params[0] = (DeviceViewable*)PyLong_AsVoidPtr(arg3);
	}
	else
	{
		if (num_params != 0)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, 0);
			Py_RETURN_NONE;
		}
	}
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 4));
	kernel->launch(gridDim, blockDim, params.data(), sharedMemBytes);
	return PyLong_FromLong(0);
}