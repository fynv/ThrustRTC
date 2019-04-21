#include <Python.h>
#include "TRTCContext.h"
#include "copy.h"

static PyObject* n_copy(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	if (TRTC_Copy(*ctx, *vec_in, *vec_out, begin_in, end_in, begin_out))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
