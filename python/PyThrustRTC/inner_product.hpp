#include <Python.h>
#include "TRTCContext.h"
#include "inner_product.h"
#include "viewbuf_to_python.hpp"
#include "functor.hpp"

static PyObject* n_inner_product(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec1 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec2 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DeviceViewable* init = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));

	PyObject* py_binary_op1 = PyTuple_GetItem(args, 4);
	bool have_op1 = false;
	Functor binary_op1;
	if (py_binary_op1 != Py_None)
	{
		binary_op1 = PyFunctor_AsFunctor(py_binary_op1);
		have_op1 = true;
	}

	PyObject* py_binary_op2 = PyTuple_GetItem(args, 5);
	bool have_op2 = false;
	Functor binary_op2;
	if (py_binary_op2 != Py_None)
	{
		binary_op2 = PyFunctor_AsFunctor(py_binary_op2);
		have_op2 = true;
	}

	size_t begin1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t end1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin2= (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));

	ViewBuf ret;
	if (!have_op1 || !have_op2)
	{
		if (TRTC_Inner_Product(*ctx, *vec1, *vec2, *init, ret, begin1, end1, begin2))
			return PyValue_FromViewBuf(ret, init->name_view_cls().c_str());
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Inner_Product(*ctx, *vec1, *vec2, *init, ret, binary_op1, binary_op2, begin1, end1, begin2))
			return PyValue_FromViewBuf(ret, init->name_view_cls().c_str());
		Py_RETURN_NONE;
	}
}

