#include <Python.h>
#include "TRTCContext.h"
#include "transform_scan.h"

static PyObject* n_transform_inclusive_scan(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	PyObject* py_unary_op = PyTuple_GetItem(args, 3);
	Functor* unary_op = (Functor*)PyLong_AsVoidPtr(py_unary_op);
	PyObject* py_binary_op = PyTuple_GetItem(args, 4);
	Functor* binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	if (TRTC_Transform_Inclusive_Scan(*ctx, *vec_in, *vec_out, *unary_op, *binary_op, begin_in, end_in, begin_out))
		return PyLong_FromLong(0);
	Py_RETURN_NONE;
}

static PyObject* n_transform_exclusive_scan(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	PyObject* py_unary_op = PyTuple_GetItem(args, 3);
	Functor* unary_op = (Functor*)PyLong_AsVoidPtr(py_unary_op);
	PyObject* pyinit = PyTuple_GetItem(args, 4);
	DeviceViewable* init = (DeviceViewable*)PyLong_AsVoidPtr(pyinit);
	PyObject* py_binary_op = PyTuple_GetItem(args, 5);
	Functor* binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));
	if (TRTC_Transform_Exclusive_Scan(*ctx, *vec_in, *vec_out, *unary_op, *init, *binary_op, begin_in, end_in, begin_out))
		return PyLong_FromLong(0);
	Py_RETURN_NONE;
}
