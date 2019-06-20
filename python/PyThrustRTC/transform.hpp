#include <Python.h>
#include "TRTCContext.h"
#include "transform.h"

static PyObject* n_transform(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* op = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	if (TRTC_Transform(*vec_in,* vec_out, *op))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_transform_binary(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in1 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in2 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* op = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	if(TRTC_Transform_Binary(*vec_in1, *vec_in2, *vec_out, *op))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_transform_if(PyObject* self, PyObject* args)
{	
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* op = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	TRTC_Transform_If(*vec_in, *vec_out, *op, *pred);
	return PyLong_FromLong(0);
}

static PyObject* n_transform_if_stencil(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_stencil = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* op = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 4));
	if(TRTC_Transform_If_Stencil(*vec_in, *vec_stencil, *vec_out, *op, *pred))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_transform_binary_if_stencil(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in1 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in2 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_stencil = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	Functor* op = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 4));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 5));
	if(TRTC_Transform_Binary_If_Stencil(*vec_in1, *vec_in2, *vec_stencil, *vec_out, *op, *pred))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

