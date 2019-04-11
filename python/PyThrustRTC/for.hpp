#include <Python.h>
#include "TRTCContext.h"
#include "for.h"

static PyObject* n_for_launch_once(PyObject* self, PyObject* args)
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
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 6));

	TRTC_For_Once(*ctx, begin, end, arg_map, name_iter, code_body, sharedMemBytes);

	return PyLong_FromLong(0);
}


static PyObject* n_for_create(PyObject* self, PyObject* args)
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
	const char* name_iter = PyUnicode_AsUTF8(PyTuple_GetItem(args, 2));
	const char* body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 3));

	TRTC_For* cptr = new TRTC_For(*ctx, params, name_iter, body);
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
	TRTC_For* cptr = (TRTC_For*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t num_params = cptr->num_params();
	size_t begin = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));
	size_t end = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 2));

	PyObject* arg3 = PyTuple_GetItem(args, 3);
	std::vector<const DeviceViewable*> params;
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
	cptr->launch(begin, end, params.data(), sharedMemBytes);
	return PyLong_FromLong(0);
}


static PyObject* n_for_template_create(PyObject* self, PyObject* args)
{
	PyObject* pyTemplParamList = PyTuple_GetItem(args, 0);
	ssize_t num_templ_params = PyList_Size(pyTemplParamList);
	std::vector<const char*> templ_params(num_templ_params);
	for (ssize_t i = 0; i < num_templ_params; i++)
		templ_params[i] = PyUnicode_AsUTF8(PyList_GetItem(pyTemplParamList, i));

	PyObject* pyParamList = PyTuple_GetItem(args, 1);
	ssize_t num_params = PyList_Size(pyParamList);
	std::vector<TRTCContext::ParamDesc> params(num_params);
	for (ssize_t i = 0; i < num_params; i++)
	{
		PyObject* tuple = PyList_GetItem(pyParamList, i);
		params[i].type = PyUnicode_AsUTF8(PyTuple_GetItem(tuple, 0));
		params[i].name = PyUnicode_AsUTF8(PyTuple_GetItem(tuple, 1));
	}
	const char* name_iter = PyUnicode_AsUTF8(PyTuple_GetItem(args, 2));
	const char* body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 3));

	TRTC_For_Template* ftempl =new TRTC_For_Template(templ_params, params, name_iter, body);
	return PyLong_FromVoidPtr((void*)ftempl);
}


static PyObject* n_for_template_destroy(PyObject* self, PyObject* args)
{
	TRTC_For_Template*  kernel = (TRTC_For_Template*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete kernel;
	return PyLong_FromLong(0);
}

static PyObject* n_for_template_num_template_params(PyObject* self, PyObject* args)
{
	TRTC_For_Template*  kernel = (TRTC_For_Template*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLong((long)kernel->num_template_params());
}

static PyObject* n_for_template_num_params(PyObject* self, PyObject* args)
{
	TRTC_For_Template*  kernel = (TRTC_For_Template*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLong((long)kernel->num_params());
}


static PyObject* n_for_template_launch_explict(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	TRTC_For_Template*  kernel = (TRTC_For_Template*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t num_template_params = kernel->num_template_params();
	size_t num_params = kernel->num_params();
	size_t begin = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 2));
	size_t end = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 3));

	PyObject* arg3 = PyTuple_GetItem(args, 4);
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

	PyObject* arg4 = PyTuple_GetItem(args, 5);
	std::vector<const DeviceViewable*> params;
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
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 6));
	kernel->launch_explict(*ctx, template_args, begin, end, params.data(), sharedMemBytes);
	return PyLong_FromLong(0);
}

static PyObject* n_for_template_launch(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	TRTC_For_Template*  kernel = (TRTC_For_Template*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t num_params = kernel->num_params();
	size_t begin = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 2));
	size_t end = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 3));

	PyObject* arg3 = PyTuple_GetItem(args, 4);
	std::vector<const DeviceViewable*> params;
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
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 5));
	if (kernel->launch(*ctx, begin, end, params.data(), sharedMemBytes))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
