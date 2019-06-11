#include <Python.h>
#include "TRTCContext.h"
#include "scan.h"

static PyObject* n_inclusive_scan(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* py_binary_op = PyTuple_GetItem(args, 2);
	Functor* binary_op = nullptr;
	if (py_binary_op != Py_None)
		binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);
	
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	if (binary_op == nullptr)
	{
		if (TRTC_Inclusive_Scan(*vec_in, *vec_out, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Inclusive_Scan(*vec_in, *vec_out, *binary_op, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
}

static PyObject* n_exclusive_scan(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* pyinit = PyTuple_GetItem(args, 2);
	DeviceViewable* init = nullptr;
	if (pyinit != Py_None)
		init = (DeviceViewable*)PyLong_AsVoidPtr(pyinit);
	PyObject* py_binary_op = PyTuple_GetItem(args, 3);
	Functor* binary_op = nullptr;
	if (py_binary_op != Py_None)
		binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);

	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	if (init == nullptr)
	{
		if (TRTC_Exclusive_Scan(*vec_in, *vec_out, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else if (binary_op == nullptr)
	{
		if (TRTC_Exclusive_Scan(*vec_in, *vec_out, *init, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Exclusive_Scan(*vec_in, *vec_out, *init, *binary_op, begin_in, end_in, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
}

static PyObject* n_inclusive_scan_by_key(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_key = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_value = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	PyObject* py_binary_pred = PyTuple_GetItem(args, 3);
	Functor* binary_pred = nullptr;
	if (py_binary_pred != Py_None)
		binary_pred = (Functor*)PyLong_AsVoidPtr(py_binary_pred);
	PyObject* py_binary_op = PyTuple_GetItem(args, 4);
	Functor* binary_op = nullptr;
	if (py_binary_op != Py_None)
		binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);
	size_t begin_key = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t end_key = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t begin_value = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));

	if (binary_pred == nullptr)
	{
		if (TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, begin_key, end_key, begin_value, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else if (binary_op == nullptr)
	{
		if (TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *binary_pred, begin_key, end_key, begin_value, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *binary_pred, *binary_op, begin_key, end_key, begin_value, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
}

static PyObject* n_exclusive_scan_by_key(PyObject* self, PyObject* args)
{
	
	DVVectorLike* vec_key = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_value = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	PyObject* pyinit = PyTuple_GetItem(args, 3);
	DeviceViewable* init = nullptr;
	if (pyinit != Py_None)
		init = (DeviceViewable*)PyLong_AsVoidPtr(pyinit);
	PyObject* py_binary_pred = PyTuple_GetItem(args, 4);
	Functor* binary_pred = nullptr;
	if (py_binary_pred != Py_None)
		binary_pred = (Functor*)PyLong_AsVoidPtr(py_binary_pred);
	PyObject* py_binary_op = PyTuple_GetItem(args, 5);
	Functor* binary_op = nullptr;
	if (py_binary_op != Py_None)
		binary_op = (Functor*)PyLong_AsVoidPtr(py_binary_op);
	size_t begin_key = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t end_key = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin_value = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 9));

	if (init == nullptr)
	{
		if (TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, begin_key, end_key, begin_value, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	if (binary_pred == nullptr)
	{
		if (TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init, begin_key, end_key, begin_value, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else if (binary_op == nullptr)
	{
		if (TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init, *binary_pred, begin_key, end_key, begin_value, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init, *binary_pred, *binary_op, begin_key, end_key, begin_value, begin_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
}

