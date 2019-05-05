#include <Python.h>
#include "TRTCContext.h"
#include "scan.h"
#include "functor.hpp"

static PyObject* n_inclusive_scan(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	PyObject* py_binary_op = PyTuple_GetItem(args, 3);
	bool have_op = false;
	Functor binary_op;
	if (py_binary_op != Py_None)
	{
		binary_op = PyFunctor_AsFunctor(py_binary_op);
		have_op = true;
	}
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	if (!have_op)
	{
		if (TRTC_Inclusive_Scan(*ctx, *vec_in, *vec_out, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Inclusive_Scan(*ctx, *vec_in, *vec_out, binary_op, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
}

static PyObject* n_exclusive_scan(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	PyObject* pyinit = PyTuple_GetItem(args, 3);
	DeviceViewable* init = nullptr;
	if (pyinit != Py_None)
		init = (DeviceViewable*)PyLong_AsVoidPtr(pyinit);
	PyObject* py_binary_op = PyTuple_GetItem(args, 4);
	bool have_op = false;
	Functor binary_op;
	if (py_binary_op != Py_None)
	{
		binary_op = PyFunctor_AsFunctor(py_binary_op);
		have_op = true;
	}
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	if (init == nullptr)
	{
		if (TRTC_Exclusive_Scan(*ctx, *vec_in, *vec_out, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else if (!have_op)
	{
		if (TRTC_Exclusive_Scan(*ctx, *vec_in, *vec_out, *init, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Exclusive_Scan(*ctx, *vec_in, *vec_out, *init, binary_op, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
}
