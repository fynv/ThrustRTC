#include <Python.h>
#include "TRTCContext.h"
#include "partition.h"

static PyObject* n_partition(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	uint32_t res = TRTC_Partition(*vec, *pred);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_partition_stencil(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* stencil = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	uint32_t res = TRTC_Partition_Stencil(*vec, *stencil, *pred);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_partition_copy(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_true = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_false = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	uint32_t res = TRTC_Partition_Copy(*vec_in, *vec_true, *vec_false, *pred);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_partition_copy_stencil(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* stencil = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_true = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVectorLike* vec_false = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 4));
	uint32_t res = TRTC_Partition_Copy_Stencil(*vec_in, *stencil, *vec_true, *vec_false, *pred);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}


static PyObject* n_partition_point(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t pp;
	if (TRTC_Partition_Point(*vec, *pred, pp))
		return PyLong_FromUnsignedLongLong((unsigned long long)pp);
	else
		Py_RETURN_NONE;
}

static PyObject* n_is_partitioned(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	bool res;
	if (TRTC_Is_Partitioned(*vec, *pred, res))
		return PyBool_FromLong(res ? 1 : 0);
	else
		Py_RETURN_NONE;

}
