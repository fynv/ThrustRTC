#include <Python.h>
#include "TRTCContext.h"
#include "reduce.h"
#include "viewbuf_to_python.hpp"

static PyObject* n_reduce(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* pyinit = PyTuple_GetItem(args, 1);
	DeviceViewable* init = nullptr;
	if (pyinit != Py_None)
		init = (DeviceViewable*)PyLong_AsVoidPtr(pyinit);
	PyObject* py_binary_op = PyTuple_GetItem(args, 2);
	Functor* binary_op = nullptr;
	if (py_binary_op != Py_None)
		binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);

	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	ViewBuf ret;
	if (init == nullptr)
	{
		if (TRTC_Reduce(*vec, ret, begin, end))
			return PyValue_FromViewBuf(ret, vec->name_elem_cls().c_str());
		Py_RETURN_NONE;
	}
	else if (binary_op == nullptr)
	{
		if (TRTC_Reduce(*vec, *init, ret, begin, end))
			return PyValue_FromViewBuf(ret, vec->name_elem_cls().c_str());
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Reduce(*vec, *init, *binary_op, ret, begin, end))
			return PyValue_FromViewBuf(ret, vec->name_elem_cls().c_str());
		Py_RETURN_NONE;
	}
}

static PyObject* n_reduce_by_key(PyObject* self, PyObject* args)
{
	DVVectorLike* key_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* value_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* key_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVectorLike* value_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	PyObject* py_binary_pred = PyTuple_GetItem(args, 4);
	Functor* binary_pred = nullptr;
	if (py_binary_pred != Py_None)
		binary_pred = (Functor*)PyLong_AsVoidPtr(py_binary_pred);
	PyObject* py_binary_op = PyTuple_GetItem(args, 5);
	Functor* binary_op = nullptr;
	if (py_binary_op != Py_None)
		binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);
	size_t begin_key_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t end_key_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin_value_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));
	size_t begin_key_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 9));
	size_t begin_value_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 10));

	if (binary_pred == nullptr)
	{
		uint32_t res = TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out, begin_key_in, end_key_in, begin_value_in, begin_key_out, begin_value_out);
		if (res != uint32_t(-1))
			return PyLong_FromUnsignedLong((unsigned long)res);
		else
			Py_RETURN_NONE;
	}
	else if (binary_op == nullptr)
	{
		uint32_t res = TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out, *binary_pred, begin_key_in, end_key_in, begin_value_in, begin_key_out, begin_value_out);
		if (res != uint32_t(-1))
			return PyLong_FromUnsignedLong((unsigned long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		uint32_t res = TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out, *binary_pred, *binary_op, begin_key_in, end_key_in, begin_value_in, begin_key_out, begin_value_out);
		if (res != uint32_t(-1))
			return PyLong_FromUnsignedLong((unsigned long)res);
		else
			Py_RETURN_NONE;
	}
}

