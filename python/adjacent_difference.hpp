#include <Python.h>
#include "TRTCContext.h"
#include "adjacent_difference.h"

static PyObject* n_adjacent_difference(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* py_binary_op = PyTuple_GetItem(args, 2);
	Functor* binary_op = nullptr;
	if (py_binary_op != Py_None)
	{
		binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);
	}

	if (binary_op == nullptr)
	{
		if (TRTC_Adjacent_Difference(*vec_in, *vec_out))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Adjacent_Difference(*vec_in, *vec_out, *binary_op))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}
