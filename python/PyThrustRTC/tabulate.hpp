#include <Python.h>
#include "TRTCContext.h"
#include "tabulate.h"

static PyObject* n_tabulate(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* op = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 2));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	if (TRTC_tabulate(*vec, *op, begin, end))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
