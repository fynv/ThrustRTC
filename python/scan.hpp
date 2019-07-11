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

	if (binary_op == nullptr)
	{
		if (TRTC_Inclusive_Scan(*vec_in, *vec_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Inclusive_Scan(*vec_in, *vec_out, *binary_op))
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

	if (init == nullptr)
	{
		if (TRTC_Exclusive_Scan(*vec_in, *vec_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else if (binary_op == nullptr)
	{
		if (TRTC_Exclusive_Scan(*vec_in, *vec_out, *init))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Exclusive_Scan(*vec_in, *vec_out, *init, *binary_op))
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

	if (binary_pred == nullptr)
	{
		if (TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else if (binary_op == nullptr)
	{
		if (TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *binary_pred))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *binary_pred, *binary_op))
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

	if (init == nullptr)
	{
		if (TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	if (binary_pred == nullptr)
	{
		if (TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else if (binary_op == nullptr)
	{
		if (TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init, *binary_pred))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init, *binary_pred, *binary_op))
			return PyLong_FromLong(0);
		Py_RETURN_NONE;
	}
}

