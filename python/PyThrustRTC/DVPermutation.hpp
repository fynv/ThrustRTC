#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVPermutation.h"

static PyObject* n_dvpermutation_create(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_value = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_index = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVPermutation* ret = new DVPermutation(*vec_value, *vec_index);
	return PyLong_FromVoidPtr(ret);
}

