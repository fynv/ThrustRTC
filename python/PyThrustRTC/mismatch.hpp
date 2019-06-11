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
	size_t begin1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t begin2 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t res1, res2;
	if (pred == nullptr)
	{
		if (TRTC_Mismatch(*vec1, *vec2, res1, res2, begin1, end1, begin2))
		{
			PyObject* ret = PyTuple_New(2);
			PyTuple_SetItem(ret, 0, PyLong_FromLongLong((long long)res1));
			PyTuple_SetItem(ret, 1, PyLong_FromLongLong((long long)res2));
			return ret;
		}
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Mismatch(*vec1, *vec2, *pred, res1, res2, begin1, end1, begin2))
		{
			PyObject* ret = PyTuple_New(2);
			PyTuple_SetItem(ret, 0, PyLong_FromLongLong((long long)res1));
			PyTuple_SetItem(ret, 1, PyLong_FromLongLong((long long)res2));
			return ret;
		}
		else
			Py_RETURN_NONE;
	}
}
