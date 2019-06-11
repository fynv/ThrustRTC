#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVCounter.h"

static PyObject* n_dvcounter_create(PyObject* self, PyObject* args)
{	
	DeviceViewable* dvobj_init = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t size = (size_t)PyLong_AsLongLong(PyTuple_GetItem(args, 1));
	DVCounter* ret = new DVCounter(*dvobj_init, size);
	return PyLong_FromVoidPtr(ret);
}

