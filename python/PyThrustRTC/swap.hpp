#include <Python.h>
#include "TRTCContext.h"
#include "swap.h"

static PyObject* n_swap(PyObject* self, PyObject* args)
{
	DVVectorLike* vec1 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec2 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	if (TRTC_Swap(*vec1, *vec2))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
