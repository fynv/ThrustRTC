#include <Python.h>
#include "TRTCContext.h"
#include "equal.h"

static PyObject* n_equal(PyObject* self, PyObject* args)
{
	DVVectorLike* vec1 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec2 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* py_binary_pred = PyTuple_GetItem(args, 2);
	Functor* binary_pred = nullptr;
	if (py_binary_pred != Py_None)
		binary_pred = (Functor*)PyLong_AsVoidPtr(py_binary_pred);

	bool res;
	if (binary_pred == nullptr)
		if (TRTC_Equal(*vec1, *vec2, res))
			return PyBool_FromLong(res ? 1 : 0);
		else
			Py_RETURN_NONE;
	else
		if (TRTC_Equal(*vec1, *vec2, *binary_pred, res))
			return PyBool_FromLong(res ? 1 : 0);
		else
			Py_RETURN_NONE;
}
