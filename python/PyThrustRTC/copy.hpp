#include <Python.h>
#include "TRTCContext.h"
#include "copy.h"

static PyObject* n_copy(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 2));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	if (TRTC_Copy(*vec_in, *vec_out, begin_in, end_in, begin_out))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_copy_if(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	uint32_t res = TRTC_Copy_If(*vec_in, *vec_out, *pred, begin_in, end_in, begin_out);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_copy_if_stencil(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_stencil = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t begin_stencil = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	uint32_t res = TRTC_Copy_If_Stencil(*vec_in, *vec_stencil, *vec_out, *pred, begin_in, end_in, begin_stencil, begin_out);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}
