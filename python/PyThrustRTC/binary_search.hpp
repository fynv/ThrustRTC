#include <Python.h>
#include "TRTCContext.h"
#include "binary_search.h"

static PyObject* n_lower_bound(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* comp = nullptr;
	PyObject* py_comp = PyTuple_GetItem(args, 2);
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	if (comp == nullptr)
	{
		size_t res;
		if (TRTC_Lower_Bound(*vec, *value, res))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		size_t res;
		if (TRTC_Lower_Bound(*vec, *value, *comp, res))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_upper_bound(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* comp = nullptr;
	PyObject* py_comp = PyTuple_GetItem(args, 2);
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	if (comp == nullptr)
	{
		size_t res;
		if (TRTC_Upper_Bound(*vec, *value, res))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		size_t res;
		if (TRTC_Upper_Bound(*vec, *value, *comp, res))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_binary_search(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* comp = nullptr;
	PyObject* py_comp = PyTuple_GetItem(args, 2);
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	if (comp == nullptr)
	{
		bool res;
		if (TRTC_Binary_Search(*vec, *value, res))
			return PyBool_FromLong(res?(long)1:(long)0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		bool res;
		if (TRTC_Binary_Search(*vec, *value, *comp, res))
			return PyBool_FromLong(res ? (long)1 : (long)0);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_lower_bound_v(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* values = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* result = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* comp = nullptr;
	PyObject* py_comp = PyTuple_GetItem(args, 3);
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	
	if (comp == nullptr)
	{
		if (TRTC_Lower_Bound_V(*vec, *values, *result))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Lower_Bound_V(*vec, *values, *result, *comp))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_upper_bound_v(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* values = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* result = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* comp = nullptr;
	PyObject* py_comp = PyTuple_GetItem(args, 3);
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	
	if (comp == nullptr)
	{
		if (TRTC_Upper_Bound_V(*vec, *values, *result))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Upper_Bound_V(*vec, *values, *result, *comp))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_binary_search_v(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* values = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* result = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* comp = nullptr;
	PyObject* py_comp = PyTuple_GetItem(args, 3);
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);

	if (comp == nullptr)
	{
		if (TRTC_Binary_Search_V(*vec, *values, *result))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Binary_Search_V(*vec, *values, *result, *comp))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}

