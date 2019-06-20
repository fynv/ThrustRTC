#include <Python.h>
#include "TRTCContext.h"
#include "fill.h"

static PyObject* n_fill(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	if(TRTC_Fill(*vec, *value))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
