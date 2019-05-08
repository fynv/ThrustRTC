#include <Python.h>
#include "TRTCContext.h"

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

static PyObject* n_context_add_built_in_header(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* filename = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	const char* filecontent = PyUnicode_AsUTF8(PyTuple_GetItem(args, 2));
	ctx->add_built_in_header(filename, filecontent);
	return PyLong_FromLong(0);
}

static PyObject* n_context_add_inlcude_filename(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* fn = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	ctx->add_inlcude_filename(fn);

	return PyLong_FromLong(0);
}

static PyObject* n_context_add_code_block(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* line = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	ctx->add_code_block(line);
	return PyLong_FromLong(0);
}

static PyObject* n_context_add_constant_object(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* name = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	DeviceViewable* dv = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	ctx->add_constant_object(name, *dv);
	return PyLong_FromLong(0);
}

static PyObject* n_context_calc_optimal_block_size(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* pyArgMap = PyTuple_GetItem(args, 1);
	ssize_t num_params = PyList_Size(pyArgMap);
	std::vector<TRTCContext::AssignedParam> arg_map(num_params);
	for (ssize_t i = 0; i < num_params; i++)
	{
		PyObject* pyAssignedParam = PyList_GetItem(pyArgMap, i);
		arg_map[i].param_name = PyUnicode_AsUTF8(PyTuple_GetItem(pyAssignedParam, 0));
		arg_map[i].arg = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(pyAssignedParam, 1));
	}
	const char* code_body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 2));
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 3));
	int sizeBlock;
	if (ctx->calc_optimal_block_size(arg_map, code_body, sizeBlock, sharedMemBytes))
		return PyLong_FromLong((long)sizeBlock);
	else
		Py_RETURN_NONE;

}

static PyObject* n_context_launch_kernel(PyObject* self, PyObject* args)
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
	if (ctx->launch_kernel(gridDim, blockDim, arg_map, code_body, sharedMemBytes))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}


static PyObject* n_context_launch_for(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t begin = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));
	size_t end = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 2));

	PyObject* pyArgMap = PyTuple_GetItem(args, 3);
	ssize_t num_params = PyList_Size(pyArgMap);
	std::vector<TRTCContext::AssignedParam> arg_map(num_params);
	for (ssize_t i = 0; i < num_params; i++)
	{
		PyObject* pyAssignedParam = PyList_GetItem(pyArgMap, i);
		arg_map[i].param_name = PyUnicode_AsUTF8(PyTuple_GetItem(pyAssignedParam, 0));
		arg_map[i].arg = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(pyAssignedParam, 1));
	}

	const char* name_iter = PyUnicode_AsUTF8(PyTuple_GetItem(args, 4));
	const char* code_body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 5));

	if(ctx->launch_for(begin, end, arg_map, name_iter, code_body))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_kernel_create(PyObject* self, PyObject* args)
{
	PyObject* pyParamList = PyTuple_GetItem(args, 0);
	ssize_t num_params = PyList_Size(pyParamList);
	std::vector<const char*> params(num_params);
	for (ssize_t i = 0; i < num_params; i++)
		params[i] = PyUnicode_AsUTF8(PyList_GetItem(pyParamList, i));		
	const char* body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	TRTC_Kernel* cptr = new TRTC_Kernel(params, body);

	return PyLong_FromVoidPtr(cptr);
}

static PyObject* n_kernel_destroy(PyObject* self, PyObject* args)
{
	TRTC_Kernel* cptr = (TRTC_Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete cptr;
	return PyLong_FromLong(0);
}

static PyObject* n_kernel_num_params(PyObject* self, PyObject* args)
{
	TRTC_Kernel* cptr = (TRTC_Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLong((long)cptr->num_params());
}

static PyObject* n_kernel_calc_optimal_block_size(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	TRTC_Kernel* cptr = (TRTC_Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t num_params = cptr->num_params();

	PyObject* arg2 = PyTuple_GetItem(args, 2);
	std::vector<const DeviceViewable*> params;
	if (PyObject_TypeCheck(arg2, &PyList_Type))
	{
		ssize_t size = PyList_Size(arg2);
		if ((ssize_t)num_params != size)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, size);
			Py_RETURN_NONE;
		}
		params.resize(size);
		for (ssize_t i = 0; i < size; i++)
			params[i] = (DeviceViewable*)PyLong_AsVoidPtr(PyList_GetItem(arg2, i));
	}
	else if (arg2 != Py_None)
	{
		if (num_params != 1)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, 1);
			Py_RETURN_NONE;
		}
		params.resize(1);
		params[0] = (DeviceViewable*)PyLong_AsVoidPtr(arg2);
	}
	else
	{
		if (num_params != 0)
		{
			PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, 0);
			Py_RETURN_NONE;
		}
	}
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 3));
	int sizeBlock;
	if (cptr->calc_optimal_block_size(*ctx, params.data(), sizeBlock, sharedMemBytes))
		return PyLong_FromLong((long)sizeBlock);
	else
		Py_RETURN_NONE;
}

static PyObject* n_kernel_launch(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	TRTC_Kernel* cptr = (TRTC_Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t num_params = cptr->num_params();

	dim_type gridDim;
	PyObject* arg2 = PyTuple_GetItem(args, 2);
	if (PyObject_TypeCheck(arg2, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg2);
		gridDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 0)) : 1;
		gridDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 1)) : 1;
		gridDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 2)) : 1;
	}
	else
	{
		gridDim.x = (unsigned)PyLong_AsUnsignedLong(arg2);
		gridDim.y = 1;
		gridDim.z = 1;
	}

	dim_type blockDim;
	PyObject* arg3 = PyTuple_GetItem(args, 3);
	if (PyObject_TypeCheck(arg3, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg3);
		blockDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg3, 0)) : 1;
		blockDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg3, 1)) : 1;
		blockDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg3, 2)) : 1;
	}
	else
	{
		blockDim.x = (unsigned)PyLong_AsUnsignedLong(arg3);
		blockDim.y = 1;
		blockDim.z = 1;
	}

	PyObject* arg4 = PyTuple_GetItem(args, 4);
	std::vector<const DeviceViewable*> params;
	if (PyObject_TypeCheck(arg4, &PyList_Type))
	{
		ssize_t size = PyList_Size(arg4);
		if ((ssize_t)num_params != size)
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
	if(cptr->launch(*ctx, gridDim, blockDim, params.data(), sharedMemBytes))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}


static PyObject* n_for_create(PyObject* self, PyObject* args)
{
	PyObject* pyParamList = PyTuple_GetItem(args, 0);
	ssize_t num_params = PyList_Size(pyParamList);
	std::vector<const char*> params(num_params);
	for (ssize_t i = 0; i < num_params; i++)
		params[i] = PyUnicode_AsUTF8(PyList_GetItem(pyParamList, i));
	const char* name_iter = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	const char* body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 2));
	TRTC_For* cptr = new TRTC_For(params, name_iter, body);

	return PyLong_FromVoidPtr(cptr);
}

static PyObject* n_for_destroy(PyObject* self, PyObject* args)
{
	TRTC_For* cptr = (TRTC_For*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete cptr;
	return PyLong_FromLong(0);
}

static PyObject* n_for_num_params(PyObject* self, PyObject* args)
{
	TRTC_For* cptr = (TRTC_For*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLong((long)cptr->num_params());
}


static PyObject* n_for_launch(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	TRTC_For* cptr = (TRTC_For*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t num_params = cptr->num_params();
	size_t begin = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 2));
	size_t end = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 3));

	PyObject* arg4 = PyTuple_GetItem(args, 4);
	std::vector<const DeviceViewable*> params;
	if (PyObject_TypeCheck(arg4, &PyList_Type))
	{
		ssize_t size = PyList_Size(arg4);
		if ((ssize_t)num_params != size)
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
	if(cptr->launch(*ctx, begin, end, params.data()))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
