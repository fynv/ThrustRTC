#include <Python.h>
#include "TRTCContext.h"
#include "mismatch.h"

static PyObject* n_mismatch(PyObject* self, PyObject* args)
{
	DVVectorLike* vec1 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec2 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* pred = nullptr;
	PyObject* PyPred = PyTuple_GetItem(args, 2);
	if (PyPred != Py_None)
		pred = (Functor*)(DVVectorLike*)PyLong_AsVoidPtr(PyPred);
	size_t res;
	if (pred == nullptr)
	{
		if (TRTC_Mismatch(*vec1, *vec2, res))
			return PyLong_FromLongLong((long long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Mismatch(*vec1, *vec2, *pred, res))
			return PyLong_FromLongLong((long long)res);
		else
			Py_RETURN_NONE;
	}
}
