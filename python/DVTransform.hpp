#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVTransform.h"

static PyObject* n_dvtransform_create(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* elem_cls = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	Functor* op = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2)); 
	DVTransform* ret = new DVTransform(*vec_in, elem_cls, *op);
	return PyLong_FromVoidPtr(ret);
}

