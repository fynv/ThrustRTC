#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVCustomVector.h"

static PyObject* n_dvcustomvector_create(PyObject* self, PyObject* args)
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
	const char* name_idx = PyUnicode_AsUTF8(PyTuple_GetItem(args, 2));
	const char* body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 3));
	const char* elem_cls = PyUnicode_AsUTF8(PyTuple_GetItem(args, 4));
	size_t size = (size_t)PyLong_AsLongLong(PyTuple_GetItem(args, 5));
	bool read_only = PyObject_IsTrue(PyTuple_GetItem(args, 6)) != 0;
	DVCustomVector* ret = new DVCustomVector(*ctx, arg_map, name_idx, body, elem_cls, size, read_only);
	return PyLong_FromVoidPtr(ret);
}
