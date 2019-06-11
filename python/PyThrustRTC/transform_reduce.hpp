#include <Python.h>
#include "TRTCContext.h"
#include "transform_reduce.h"
#include "viewbuf_to_python.hpp"

static PyObject* n_transform_reduce(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* unary_op = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DeviceViewable* init = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* binary_op = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	ViewBuf ret;
	if (TRTC_Transform_Reduce(*vec, *unary_op, *init, *binary_op, ret, begin, end))
		return PyValue_FromViewBuf(ret, init->name_view_cls().c_str());
	Py_RETURN_NONE;
}
