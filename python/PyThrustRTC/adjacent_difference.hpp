#include <Python.h>
#include "TRTCContext.h"
#include "adjacent_difference.h"

static PyObject* n_adjacent_difference(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	PyObject* py_binary_op = PyTuple_GetItem(args, 3);
	Functor* binary_op = nullptr;
	if (py_binary_op != Py_None)
	{
		binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);
	}
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));

	if (binary_op == nullptr)
	{
		if (TRTC_Adjacent_Difference(*ctx, *vec_in, *vec_out, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Adjacent_Difference(*ctx, *vec_in, *vec_out, *binary_op, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}
