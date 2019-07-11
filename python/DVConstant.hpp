#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVConstant.h"

static PyObject* n_dvconstant_create(PyObject* self, PyObject* args)
{
	DeviceViewable* dvobj = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t size = (size_t)PyLong_AsLongLong(PyTuple_GetItem(args, 1));
	DVConstant* ret = new DVConstant(*dvobj, size);
	return PyLong_FromVoidPtr(ret);
}

