#include <Python.h>
#include "TRTCContext.h"
#include "partition.h"

static PyObject* n_partition(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	uint32_t res = TRTC_Partition(*ctx, *vec, *pred, begin, end);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_partition_stencil(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* stencil = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t begin_stencil = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	uint32_t res = TRTC_Partition_Stencil(*ctx, *vec, *stencil, *pred, begin, end, begin_stencil);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_partition_copy(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_true = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVectorLike* vec_false = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 4));
	size_t begin_in  = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t begin_true = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin_false = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));
	uint32_t res = TRTC_Partition_Copy(*ctx, *vec_in, *vec_true, *vec_false, *pred, begin_in, end_in, begin_true, begin_false);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_partition_copy_stencil(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* stencil = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVectorLike* vec_true = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	DVVectorLike* vec_false = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 4));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 5));
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin_stencil = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));
	size_t begin_true = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 9));
	size_t begin_false = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 10));
	uint32_t res = TRTC_Partition_Copy_Stencil(*ctx, *vec_in, *stencil, *vec_true, *vec_false, *pred, begin_in, end_in, begin_stencil, begin_true, begin_false);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}


static PyObject* n_partition_point(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t pp;
	if (TRTC_Partition_Point(*ctx, *vec, *pred, pp, begin, end))
		return PyLong_FromUnsignedLongLong((unsigned long long)pp);
	else
		Py_RETURN_NONE;
}
