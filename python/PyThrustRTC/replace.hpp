#include <Python.h>
#include "TRTCContext.h"
#include "replace.h"
#include "functor.hpp"

static PyObject* n_replace(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DeviceViewable* old_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DeviceViewable* new_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	if (TRTC_Replace(*ctx, *vec, *old_value, *new_value, begin, end))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_replace_if(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor pred = PyFunctor_AsFunctor(PyTuple_GetItem(args, 2));
	DeviceViewable* new_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	if (TRTC_Replace_If(*ctx, *vec, pred, *new_value, begin, end))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_replace_copy(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DeviceViewable* old_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	DeviceViewable* new_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 4));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	if (TRTC_Replace_Copy(*ctx, *vec_in, *vec_out, *old_value, *new_value, begin_in, end_in, begin_out))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_replace_copy_if(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor pred = PyFunctor_AsFunctor(PyTuple_GetItem(args, 3));
	DeviceViewable* new_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 4));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	if(TRTC_Replace_Copy_If(*ctx, *vec_in, *vec_out, pred, *new_value, begin_in, end_in, begin_out))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
