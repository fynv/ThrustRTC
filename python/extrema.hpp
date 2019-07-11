#include <Python.h>
#include "TRTCContext.h"
#include "extrema.h"

static PyObject* n_min_element(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* py_comp = PyTuple_GetItem(args, 1);
	Functor* comp = nullptr;
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);

	size_t id_min;
	if (comp==nullptr)
		if (TRTC_Min_Element(*vec, id_min))
			return PyLong_FromUnsignedLongLong((unsigned long long)id_min);
		else
			Py_RETURN_NONE;
	else
		if (TRTC_Min_Element(*vec, *comp, id_min))
			return PyLong_FromUnsignedLongLong((unsigned long long)id_min);
		else
			Py_RETURN_NONE;
}


static PyObject* n_max_element(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* py_comp = PyTuple_GetItem(args, 1);
	Functor* comp = nullptr;
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);

	size_t id_max;
	if (comp == nullptr)
		if (TRTC_Max_Element(*vec, id_max))
			return PyLong_FromUnsignedLongLong((unsigned long long)id_max);
		else
			Py_RETURN_NONE;
	else
		if (TRTC_Max_Element(*vec, *comp, id_max))
			return PyLong_FromUnsignedLongLong((unsigned long long)id_max);
		else
			Py_RETURN_NONE;
}


static PyObject* n_minmax_element(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* py_comp = PyTuple_GetItem(args, 1);
	Functor* comp = nullptr;
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);

	size_t id_min, id_max;
	if (comp == nullptr)
	{
		if (!TRTC_MinMax_Element(*vec, id_min, id_max))
			Py_RETURN_NONE;
	}
	else
	{
		if (!TRTC_MinMax_Element(*vec, *comp, id_min, id_max))
			Py_RETURN_NONE;
	}

	PyObject* ret = PyTuple_New(2);
	PyTuple_SetItem(ret, 0, PyLong_FromUnsignedLongLong((unsigned long long)id_min));
	PyTuple_SetItem(ret, 1, PyLong_FromUnsignedLongLong((unsigned long long)id_max));
	return ret;
}
