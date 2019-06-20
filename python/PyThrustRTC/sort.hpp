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

	if (comp == nullptr)
	{
		if (TRTC_Sort(*vec))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Sort(*vec, *comp))
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
	
	if (comp == nullptr)
	{
		if (TRTC_Sort_By_Key(*keys, *values))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Sort_By_Key(*keys, *values, *comp))
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
	bool res;
	if (comp == nullptr)
	{
		if (TRTC_Is_Sorted(*vec, res))
			return PyBool_FromLong(res ? 1 : 0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Is_Sorted(*vec, *comp, res))
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
	size_t res;
	if (comp == nullptr)
	{
		if (TRTC_Is_Sorted_Until(*vec, res))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Is_Sorted_Until(*vec, *comp, res))
			return PyLong_FromUnsignedLongLong((unsigned long long)res);
		else
			Py_RETURN_NONE;
	}
}

