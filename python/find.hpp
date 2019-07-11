#include <Python.h>
#include "TRTCContext.h"
#include "find.h"

static PyObject* n_find(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t res;
	if (TRTC_Find(*vec, *value, res))
		return PyLong_FromLongLong((long long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_find_if(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t res;
	if (TRTC_Find_If(*vec, *pred, res))
		return PyLong_FromLongLong((long long)res);
	else
		Py_RETURN_NONE;
}


static PyObject* n_find_if_not(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t res;
	if (TRTC_Find_If_Not(*vec, *pred, res))
		return PyLong_FromLongLong((long long)res);
	else
		Py_RETURN_NONE;
}

