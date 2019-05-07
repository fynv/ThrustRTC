#include <Python.h>
#include "TRTCContext.h"
#include "reduce.h"
#include "viewbuf_to_python.hpp"

static PyObject* n_reduce(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* pyinit = PyTuple_GetItem(args, 2);
	DeviceViewable* init = nullptr;
	if (pyinit != Py_None)
		init = (DeviceViewable*)PyLong_AsVoidPtr(pyinit);
	PyObject* py_binary_op = PyTuple_GetItem(args, 3);
	Functor* binary_op = nullptr;
	if (py_binary_op != Py_None)
		binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);

	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	ViewBuf ret;
	if (init == nullptr)
	{
		if (TRTC_Reduce(*ctx, *vec, ret, begin, end))
			return PyValue_FromViewBuf(ret, vec->name_elem_cls().c_str());
		Py_RETURN_NONE;
	}
	else if (binary_op == nullptr)
	{
		if (TRTC_Reduce(*ctx, *vec, *init, ret, begin, end))
			return PyValue_FromViewBuf(ret, vec->name_elem_cls().c_str());
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Reduce(*ctx, *vec, *init, *binary_op, ret, begin, end))
			return PyValue_FromViewBuf(ret, vec->name_elem_cls().c_str());
		Py_RETURN_NONE;
	}
}
