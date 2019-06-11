#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVReverse.h"

static PyObject* n_dvreverse_create(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_value = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVReverse* ret = new DVReverse(*vec_value);
	return PyLong_FromVoidPtr(ret);
}

