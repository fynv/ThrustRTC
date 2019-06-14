#include <Python.h>
#include "TRTCContext.h"
#include "sequence.h"

static PyObject* n_sequence(PyObject* self, PyObject* args)
{	
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject* py_value_init = PyTuple_GetItem(args, 1);
	DeviceViewable* value_init = nullptr;
	if (py_value_init!=Py_None)
		value_init = (DeviceViewable*)PyLong_AsVoidPtr(py_value_init);
	PyObject* py_value_step = PyTuple_GetItem(args, 2);
	DeviceViewable* value_step = nullptr;
	if (py_value_step != Py_None)
		value_step = (DeviceViewable*)PyLong_AsVoidPtr(py_value_step);
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));

	if (value_init == nullptr)
	{
		if (TRTC_Sequence(*vec, begin, end))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else if (value_step == nullptr)
	{
		if (TRTC_Sequence(*vec, *value_init, begin, end))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Sequence(*vec, *value_init, *value_step, begin, end))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}
