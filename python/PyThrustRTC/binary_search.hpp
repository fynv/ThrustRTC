#include <Python.h>
#include "TRTCContext.h"
#include "binary_search.h"

static PyObject* n_lower_bound(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* comp = nullptr;
	PyObject* py_comp = PyTuple_GetItem(args, 3);
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	if (comp == nullptr)
	{
		size_t res;
		if (TRTC_Lower_Bound(*ctx, *vec, *value, res, begin, end))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		size_t res;
		if (TRTC_Lower_Bound(*ctx, *vec, *value, *comp, res, begin, end))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_upper_bound(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* comp = nullptr;
	PyObject* py_comp = PyTuple_GetItem(args, 3);
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	if (comp == nullptr)
	{
		size_t res;
		if (TRTC_Upper_Bound(*ctx, *vec, *value, res, begin, end))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		size_t res;
		if (TRTC_Upper_Bound(*ctx, *vec, *value, *comp, res, begin, end))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_binary_search(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* comp = nullptr;
	PyObject* py_comp = PyTuple_GetItem(args, 3);
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	if (comp == nullptr)
	{
		bool res;
		if (TRTC_Binary_Search(*ctx, *vec, *value, res, begin, end))
			return PyBool_FromLong(res?(long)1:(long)0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		bool res;
		if (TRTC_Binary_Search(*ctx, *vec, *value, *comp, res, begin, end))
			return PyBool_FromLong(res ? (long)1 : (long)0);
		else
			Py_RETURN_NONE;
	}
}
