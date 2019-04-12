#include <Python.h>
#include "TRTCContext.h"
#include "fill.h"

static PyObject* n_fill(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVector* vec = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	if(TRTC_Fill(*ctx, *vec, *value, begin, end))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
