#include <Python.h>
#include "TRTCContext.h"
#include "gather.h"

static PyObject* n_gather(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_map = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	size_t begin_map = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end_map = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	if (TRTC_Gather(*ctx, *vec_map, *vec_in, *vec_out, begin_map, end_map, begin_in, begin_out))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_gather_if(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_map = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_stencil = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 4));
	PyObject *py_pred = PyTuple_GetItem(args, 5);
	bool has_pred = false;
	Functor pred;
	if (py_pred != Py_None)
	{
		has_pred = true;
		pred = PyFunctor_AsFunctor(py_pred);
	}
	size_t begin_map = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t end_map = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin_stencil = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 9));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 10));
	if (!has_pred)
	{
		if (TRTC_Gather_If(*ctx, *vec_map, *vec_stencil, *vec_in, *vec_out, begin_map, end_map, begin_stencil, begin_in, begin_out))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Gather_If(*ctx, *vec_map, *vec_stencil, *vec_in, *vec_out, pred, begin_map, end_map, begin_stencil, begin_in, begin_out))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}
