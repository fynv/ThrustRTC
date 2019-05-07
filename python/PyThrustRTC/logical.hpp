#include <Python.h>
#include "TRTCContext.h"
#include "logical.h"

static PyObject* n_all_of(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	bool res;
	if (TRTC_All_Of(*ctx, *vec, *pred, res, begin, end))
		return PyBool_FromLong(res ? 1 : 0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_any_of(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	bool res;
	if (TRTC_Any_Of(*ctx, *vec, *pred, res, begin, end))
		return PyBool_FromLong(res ? 1 : 0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_none_of(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	bool res;
	if (TRTC_None_Of(*ctx, *vec, *pred, res, begin, end))
		return PyBool_FromLong(res ? 1 : 0);
	else
		Py_RETURN_NONE;
}
