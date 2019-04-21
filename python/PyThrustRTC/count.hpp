#include <Python.h>
#include "TRTCContext.h"
#include "count.h"

static PyObject* n_count(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t res = TRTC_Count(*ctx, *vec, *value, begin, end);
	if (res==(size_t)(-1)) Py_RETURN_NONE;
	return PyLong_FromUnsignedLongLong((unsigned long long)res);
}

static PyObject* n_count_if(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor pred = PyFunctor_AsFunctor(PyTuple_GetItem(args, 2));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t res = TRTC_Count_If(*ctx, *vec, pred, begin, end);
	if (res == (size_t)(-1)) Py_RETURN_NONE;
	return PyLong_FromUnsignedLongLong((unsigned long long)res);
}
