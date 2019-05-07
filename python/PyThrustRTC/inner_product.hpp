#include <Python.h>
#include "TRTCContext.h"
#include "inner_product.h"
#include "viewbuf_to_python.hpp"

static PyObject* n_inner_product(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec1 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec2 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DeviceViewable* init = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));

	PyObject* py_binary_op1 = PyTuple_GetItem(args, 4);
	Functor* binary_op1 = nullptr;
	if (py_binary_op1 != Py_None)
		binary_op1 = (Functor*)PyLong_AsVoidPtr(py_binary_op1);

	PyObject* py_binary_op2 = PyTuple_GetItem(args, 5);
	Functor* binary_op2 = nullptr;
	if (py_binary_op2 != Py_None)
		binary_op2 = (Functor*)PyLong_AsVoidPtr(py_binary_op2);

	size_t begin1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t end1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin2= (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));

	ViewBuf ret;
	if (binary_op1==nullptr || binary_op2 == nullptr)
	{
		if (TRTC_Inner_Product(*ctx, *vec1, *vec2, *init, ret, begin1, end1, begin2))
			return PyValue_FromViewBuf(ret, init->name_view_cls().c_str());
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Inner_Product(*ctx, *vec1, *vec2, *init, ret, *binary_op1, *binary_op2, begin1, end1, begin2))
			return PyValue_FromViewBuf(ret, init->name_view_cls().c_str());
		Py_RETURN_NONE;
	}
}

