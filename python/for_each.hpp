#include <Python.h>
#include "TRTCContext.h"
#include "for_each.h"

static PyObject* n_for_each(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* f = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	if (TRTC_For_Each(*vec, *f))
		return PyLong_FromLong(0);
	else 
		Py_RETURN_NONE;
}
