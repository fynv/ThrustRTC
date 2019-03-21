#include <Python.h>
#include <cuda_runtime.h>
#include <string>
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "CachedKernelTemplate.h"

static PyObject* n_set_ptx_cache(PyObject* self, PyObject* args)
{
	const char* path = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	TRTCContext::set_ptx_cache(path);
	return PyLong_FromLong(0);
}

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
		arg_map[i].param_name =  PyUnicode_AsUTF8(PyTuple_GetItem(pyAssignedParam, 0));
		arg_map[i].arg = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(pyAssignedParam, 1));
	}

	const char* code_body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 4));
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 5));
	ctx->launch_once(gridDim, blockDim, arg_map, code_body, sharedMemBytes);

	return PyLong_FromLong(0);
}

static PyObject* n_dv_name_view_cls(PyObject* self, PyObject* args)
{
	DeviceViewable* dv = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyUnicode_FromString(dv->name_view_cls().c_str());
}

static PyObject* n_dv_destroy(PyObject* self, PyObject* args)
{
	DeviceViewable* dv = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete dv;
	return PyLong_FromLong(0);
}

static PyObject* n_dv_create_basic(PyObject* self, PyObject* args)
{
	std::string type = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	PyObject* value = PyTuple_GetItem(args, 1);
	DeviceViewable* ret = nullptr;
	if (type == "int8_t")
		ret = new DVInt8((int8_t)PyLong_AsLong(value));
	else if (type == "uint8_t")
		ret = new DVUInt8((uint8_t)PyLong_AsLong(value));
	else if (type == "int16_t")
		ret = new DVInt16((int16_t)PyLong_AsLong(value));
	else if (type == "uint16_t")
		ret = new DVUInt16((uint16_t)PyLong_AsUnsignedLong(value));
	else if (type == "int32_t")
		ret = new DVInt32((int32_t)PyLong_AsLong(value));
	else if (type == "uint32_t")
		ret = new DVUInt32((uint32_t)PyLong_AsUnsignedLong(value));
	else if (type == "int64_t")
		ret = new DVInt64((int64_t)PyLong_AsLong(value));
	else if (type == "uint64_t")
		ret = new DVUInt64((uint64_t)PyLong_AsUnsignedLong(value));
	else if (type == "float")
		ret = new DVFloat((float)PyFloat_AsDouble(value));
	else if (type == "double")
		ret = new DVDouble((double)PyFloat_AsDouble(value));
	else if (type == "bool")
		ret = new DVBool(PyObject_IsTrue(value)!=0);
	return PyLong_FromVoidPtr(ret);
}

static PyObject* n_dvvector_create(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* elem_cls = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	size_t size = (size_t)PyLong_AsUnsignedLongLong(PyTuple_GetItem(args, 2));
	PyObject* py_data = PyTuple_GetItem(args, 3);
	DVVector* ret = nullptr;
	if (py_data == Py_None)
		ret = new DVVector(*ctx, elem_cls, size);
	else
	{
		void* data = PyLong_AsVoidPtr(py_data);
		ret = new DVVector(*ctx, elem_cls, size, data);
	}
	return PyLong_FromVoidPtr(ret);
}

static PyObject* n_dvvector_name_elem_cls(PyObject* self, PyObject* args)
{
	DVVector* dvvec = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyUnicode_FromString(dvvec->name_elem_cls().c_str());
}

static PyObject* n_dvvector_size(PyObject* self, PyObject* args)
{
	DVVector* dvvec = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromUnsignedLong((unsigned long)dvvec->size());
}

static PyObject* n_dvvector_to_host(PyObject* self, PyObject* args)
{
	DVVector* dvvec = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	void* ptr = PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	dvvec->to_host(ptr);
	return PyLong_FromLong(0);
}

static PyObject* n_dvvector_from_dvs(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* lst = PyTuple_GetItem(args, 1);
	ssize_t num_items = PyList_Size(lst);
	if (num_items < 1) Py_RETURN_NONE;
	PyObject* py_viewable = PyList_GetItem(lst, 0);
	DeviceViewable* dv = (DeviceViewable*)PyLong_AsVoidPtr(py_viewable);
	std::string elem_cls = dv->name_view_cls();
	for (ssize_t i = 1; i < num_items; i++)
	{
		py_viewable = PyList_GetItem(lst, i);
		dv = (DeviceViewable*)PyLong_AsVoidPtr(py_viewable);
		if (elem_cls != dv->name_view_cls())
			Py_RETURN_NONE;
	}
	size_t elem_size = ctx->size_of(elem_cls.c_str());
	std::vector<char> buf(elem_size*num_items);
	for (ssize_t i = 0; i < num_items; i++)
	{
		py_viewable = PyList_GetItem(lst, i);
		dv = (DeviceViewable*)PyLong_AsVoidPtr(py_viewable);
		memcpy(buf.data() + elem_size*i, dv->view().data(), elem_size);
	}
	DVVector* ret = new DVVector(*ctx, elem_cls.data(), num_items, buf.data());
	return PyLong_FromVoidPtr(ret);
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
	else if (arg3!= Py_None)
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

static PyMethodDef s_Methods[] = {
	{ "n_set_ptx_cache", n_set_ptx_cache, METH_VARARGS, "" },
	{ "n_context_create", n_context_create, METH_VARARGS, "" },
	{ "n_context_destroy", n_context_destroy, METH_VARARGS, "" },
	{ "n_context_set_verbose", n_context_set_verbose, METH_VARARGS, "" },
	{ "n_context_add_include_dir", n_context_add_include_dir, METH_VARARGS, "" },
	{ "n_context_add_inlcude_filename", n_context_add_inlcude_filename, METH_VARARGS, "" },
	{ "n_context_add_preprocessor", n_context_add_preprocessor, METH_VARARGS, "" },
	{ "n_context_launch_once", n_context_launch_once, METH_VARARGS, "" },
	{ "n_dv_name_view_cls", n_dv_name_view_cls, METH_VARARGS, "" },
	{ "n_dv_destroy", n_dv_destroy, METH_VARARGS, "" },
	{ "n_dv_create_basic", n_dv_create_basic, METH_VARARGS, "" },
	{ "n_dvvector_create", n_dvvector_create, METH_VARARGS, "" },
	{ "n_dvvector_name_elem_cls", n_dvvector_name_elem_cls, METH_VARARGS, "" },
	{ "n_dvvector_size", n_dvvector_size, METH_VARARGS, "" },
	{ "n_dvvector_to_host", n_dvvector_to_host, METH_VARARGS, "" },
	{ "n_dvvector_from_dvs", n_dvvector_from_dvs, METH_VARARGS, "" },
	{ "n_kernel_create", n_kernel_create, METH_VARARGS, "" },
	{ "n_kernel_destroy", n_kernel_destroy, METH_VARARGS, "" },
	{ "n_kernel_num_params", n_kernel_num_params, METH_VARARGS, "" },
	{ "n_kernel_launch", n_kernel_launch, METH_VARARGS, "" },
	{ "n_kernel_template_create", n_kernel_template_create, METH_VARARGS, "" },
	{ "n_kernel_template_destroy", n_kernel_template_destroy, METH_VARARGS, "" },
	{ "n_kernel_template_num_template_params", n_kernel_template_num_template_params, METH_VARARGS, "" },
	{ "n_kernel_template_num_params", n_kernel_template_num_params, METH_VARARGS, "" },
	{ "n_kernel_template_launch_explict", n_kernel_template_launch_explict, METH_VARARGS, "" },
	{ "n_kernel_template_launch", n_kernel_template_launch, METH_VARARGS, "" },
	0
};

static struct PyModuleDef cModPyDem = { PyModuleDef_HEAD_INIT, "ThrustRTC_module", "", -1, s_Methods };

PyMODINIT_FUNC PyInit_PyThrustRTC(void)
{
	cudaFree(0);
	return PyModule_Create(&cModPyDem);
}
