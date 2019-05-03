#include <Python.h>
#include "TRTCContext.h"
#include "extrema.h"
#include "functor.hpp"

static PyObject* n_min_element(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* py_comp = PyTuple_GetItem(args, 2);
	bool have_comp = false;
	Functor comp;
	if (py_comp != Py_None)
	{
		comp = PyFunctor_AsFunctor(py_comp);
		have_comp = true;
	}
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t id_min;
	if (!have_comp)
		if (TRTC_Min_Element(*ctx, *vec, id_min, begin, end))
			return PyLong_FromUnsignedLongLong((unsigned long long)id_min);
		else
			Py_RETURN_NONE;
	else
		if (TRTC_Min_Element(*ctx, *vec, comp, id_min, begin, end))
			return PyLong_FromUnsignedLongLong((unsigned long long)id_min);
		else
			Py_RETURN_NONE;
}


static PyObject* n_max_element(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* py_comp = PyTuple_GetItem(args, 2);
	bool have_comp = false;
	Functor comp;
	if (py_comp != Py_None)
	{
		comp = PyFunctor_AsFunctor(py_comp);
		have_comp = true;
	}
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t id_max;
	if (!have_comp)
		if (TRTC_Max_Element(*ctx, *vec, id_max, begin, end))
			return PyLong_FromUnsignedLongLong((unsigned long long)id_max);
		else
			Py_RETURN_NONE;
	else
		if (TRTC_Max_Element(*ctx, *vec, comp, id_max, begin, end))
			return PyLong_FromUnsignedLongLong((unsigned long long)id_max);
		else
			Py_RETURN_NONE;
}


static PyObject* n_minmax_element(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject* py_comp = PyTuple_GetItem(args, 2);
	bool have_comp = false;
	Functor comp;
	if (py_comp != Py_None)
	{
		comp = PyFunctor_AsFunctor(py_comp);
		have_comp = true;
	}
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t id_min, id_max;
	if (!have_comp)
	{
		if (!TRTC_MinMax_Element(*ctx, *vec, id_min, id_max, begin, end))
			Py_RETURN_NONE;
	}
	else
	{
		if (!TRTC_MinMax_Element(*ctx, *vec, comp, id_min, id_max, begin, end))
			Py_RETURN_NONE;
	}

	PyObject* ret = PyTuple_New(2);
	PyTuple_SetItem(ret, 0, PyLong_FromUnsignedLongLong((unsigned long long)id_min));
	PyTuple_SetItem(ret, 1, PyLong_FromUnsignedLongLong((unsigned long long)id_max));
	return ret;
}
