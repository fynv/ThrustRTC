#include <Python.h>
#include "TRTCContext.h"
#include "tabulate.h"

static PyObject* n_tabulate(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* op = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	if (TRTC_Tabulate(*vec, *op))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
