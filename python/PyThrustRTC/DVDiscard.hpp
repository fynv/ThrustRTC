#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVDiscard.h"

static PyObject* n_dvdiscard_create(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* elem_cls = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	size_t size = (size_t)PyLong_AsLongLong(PyTuple_GetItem(args, 2));
	DVDiscard* ret = new DVDiscard(*ctx, elem_cls, size);
	return PyLong_FromVoidPtr(ret);
}

