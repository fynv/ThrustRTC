#include <Python.h>
#include "TRTCContext.h"
#include "transform.h"
#include "functor.hpp"

static PyObject* n_transform(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVector* vec_in = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVector* vec_out = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor op = PyFunctor_AsFunctor(PyTuple_GetItem(args, 3));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	if (TRTC_Transform(*ctx, *vec_in,* vec_out, op, begin_in, end_in, begin_out))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_transform_binary(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVector* vec_in1 = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVector* vec_in2 = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVector* vec_out = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	Functor op = PyFunctor_AsFunctor(PyTuple_GetItem(args, 4));
	size_t begin_in1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t end_in1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t begin_in2 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));
	if(TRTC_Transform_Binary(*ctx, *vec_in1, *vec_in2, *vec_out, op, begin_in1, end_in1, begin_in2, begin_out))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_transform_if(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVector* vec_in = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVector* vec_out = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor op = PyFunctor_AsFunctor(PyTuple_GetItem(args, 3));
	Functor pred = PyFunctor_AsFunctor(PyTuple_GetItem(args, 4));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	TRTC_Transform_If(*ctx, *vec_in, *vec_out, op, pred, begin_in, end_in, begin_out);
	return PyLong_FromLong(0);
}

static PyObject* n_transform_if_stencil(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVector* vec_in = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVector* vec_stencil = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVector* vec_out = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	Functor op = PyFunctor_AsFunctor(PyTuple_GetItem(args, 4));
	Functor pred = PyFunctor_AsFunctor(PyTuple_GetItem(args, 5));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin_stencil = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 9));
	if(TRTC_Transform_If_Stencil(*ctx, *vec_in, *vec_stencil, *vec_out, op, pred, begin_in, end_in, begin_stencil, begin_out))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_transform_binary_if_stencil(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVector* vec_in1 = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVector* vec_in2 = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVector* vec_stencil = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	DVVector* vec_out = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 4));
	Functor op = PyFunctor_AsFunctor(PyTuple_GetItem(args, 5));
	Functor pred = PyFunctor_AsFunctor(PyTuple_GetItem(args, 6));
	size_t begin_in1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t end_in1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));
	size_t begin_in2 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 9));
	size_t begin_stencil = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 10));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 11));
	if(TRTC_Transform_Binary_If_Stencil(*ctx, *vec_in1, *vec_in2, *vec_stencil, *vec_out, op, pred, begin_in1, end_in1, begin_in2, begin_stencil, begin_out))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

