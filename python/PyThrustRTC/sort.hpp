#include <Python.h>
#include "TRTCContext.h"
#include "sort.h"

static PyObject* n_sort(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* py_comp = PyTuple_GetItem(args, 2);
	Functor* comp = nullptr;
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));

	if (comp == nullptr)
	{
		if (TRTC_Sort(*ctx, *vec, begin, end))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Sort(*ctx, *vec, *comp, begin, end))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_sort_by_key(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* keys = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* values = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	PyObject* py_comp = PyTuple_GetItem(args, 3);
	Functor* comp = nullptr;
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	size_t begin_keys = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end_keys = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t begin_values = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	
	if (comp == nullptr)
	{
		if (TRTC_Sort_By_Key(*ctx, *keys, *values, begin_keys, end_keys, begin_values))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Sort_By_Key(*ctx, *keys, *values, *comp, begin_keys, end_keys, begin_values))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}
