#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVCustomVector.h"

static PyObject* n_dvcustomvector_create(PyObject* self, PyObject* args)
{
	PyObject* pyArgMap = PyTuple_GetItem(args, 0);
	ssize_t num_params = PyList_Size(pyArgMap);
	std::vector<CapturedDeviceViewable> arg_map(num_params);
	for (ssize_t i = 0; i < num_params; i++)
	{
		PyObject* pyCapturedDeviceViewable = PyList_GetItem(pyArgMap, i);
		arg_map[i].obj_name = PyUnicode_AsUTF8(PyTuple_GetItem(pyCapturedDeviceViewable, 0));
		arg_map[i].obj = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(pyCapturedDeviceViewable, 1));
	}
	const char* name_idx = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	const char* body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 2));
	const char* elem_cls = PyUnicode_AsUTF8(PyTuple_GetItem(args, 3));
	size_t size = (size_t)PyLong_AsLongLong(PyTuple_GetItem(args, 4));
	bool read_only = PyObject_IsTrue(PyTuple_GetItem(args, 5)) != 0;
	DVCustomVector* ret = new DVCustomVector(arg_map, name_idx, body, elem_cls, size, read_only);
	return PyLong_FromVoidPtr(ret);
}
