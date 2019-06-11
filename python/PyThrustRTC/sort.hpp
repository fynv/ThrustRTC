#include <Python.h>
#include "TRTCContext.h"
#include "sort.h"

static PyObject* n_sort(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* py_comp = PyTuple_GetItem(args, 1);
	Functor* comp = nullptr;
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 2));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));

	if (comp == nullptr)
	{
		if (TRTC_Sort(*vec, begin, end))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Sort(*vec, *comp, begin, end))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_sort_by_key(PyObject* self, PyObject* args)
{
	DVVectorLike* keys = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* values = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* py_comp = PyTuple_GetItem(args, 2);
	Functor* comp = nullptr;
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	size_t begin_keys = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end_keys = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t begin_values = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	
	if (comp == nullptr)
	{
		if (TRTC_Sort_By_Key(*keys, *values, begin_keys, end_keys, begin_values))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Sort_By_Key(*keys, *values, *comp, begin_keys, end_keys, begin_values))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}


static PyObject* n_is_sorted(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* py_comp = PyTuple_GetItem(args, 1);
	Functor* comp = nullptr;
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 2));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	bool res;
	if (comp == nullptr)
	{
		if (TRTC_Is_Sorted(*vec, res, begin, end))
			return PyBool_FromLong(res ? 1 : 0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Is_Sorted(*vec, *comp, res, begin, end))
			return PyBool_FromLong(res ? 1 : 0);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_is_sorted_until(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* py_comp = PyTuple_GetItem(args, 1);
	Functor* comp = nullptr;
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 2));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t res;
	if (comp == nullptr)
	{
		if (TRTC_Is_Sorted_Until(*vec, res, begin, end))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Is_Sorted_Until(*vec, *comp, res, begin, end))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
}

