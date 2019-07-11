#include <Python.h>
#include "TRTCContext.h"
#include "transform_scan.h"

static PyObject* n_transform_inclusive_scan(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* py_unary_op = PyTuple_GetItem(args, 2);
	Functor* unary_op = (Functor*)PyLong_AsVoidPtr(py_unary_op);
	PyObject* py_binary_op = PyTuple_GetItem(args, 3);
	Functor* binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);
	if (TRTC_Transform_Inclusive_Scan(*vec_in, *vec_out, *unary_op, *binary_op))
		return PyLong_FromLong(0);
	Py_RETURN_NONE;
}

static PyObject* n_transform_exclusive_scan(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* py_unary_op = PyTuple_GetItem(args, 2);
	Functor* unary_op = (Functor*)PyLong_AsVoidPtr(py_unary_op);
	PyObject* pyinit = PyTuple_GetItem(args, 3);
	DeviceViewable* init = (DeviceViewable*)PyLong_AsVoidPtr(pyinit);
	PyObject* py_binary_op = PyTuple_GetItem(args, 4);
	Functor* binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);
	if (TRTC_Transform_Exclusive_Scan(*vec_in, *vec_out, *unary_op, *init, *binary_op))
		return PyLong_FromLong(0);
	Py_RETURN_NONE;
}
