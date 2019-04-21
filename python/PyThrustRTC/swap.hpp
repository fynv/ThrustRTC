#include <Python.h>
#include "TRTCContext.h"
#include "swap.h"

static PyObject* n_swap(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec1 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec2 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	size_t begin1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t begin2 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	if (TRTC_Swap(*ctx, *vec1, *vec2, begin1, end1, begin2))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
