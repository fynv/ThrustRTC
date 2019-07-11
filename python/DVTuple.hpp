#include <Python.h>
#include "TRTCContext.h"
#include "DVTuple.h"

static PyObject* n_dvtuple_create(PyObject* self, PyObject* args)
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
	return PyLong_FromVoidPtr(new DVTuple(arg_map));
}

