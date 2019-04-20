#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVTransform.h"
#include "functor.hpp"

static PyObject* n_dvtransform_create(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	const char* elem_cls = PyUnicode_AsUTF8(PyTuple_GetItem(args, 2));
	Functor op = PyFunctor_AsFunctor(PyTuple_GetItem(args, 3));
	DVTransform* ret = new DVTransform(*ctx, *vec_in, elem_cls, op);
	return PyLong_FromVoidPtr(ret);
}

