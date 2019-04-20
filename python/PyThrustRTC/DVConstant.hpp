#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVConstant.h"

static PyObject* n_dvconstant_create(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DeviceViewable* dvobj = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t size = (size_t)PyLong_AsLongLong(PyTuple_GetItem(args, 2));
	DVConstant* ret = new DVConstant(*ctx, *dvobj, size);
	return PyLong_FromVoidPtr(ret);
}

