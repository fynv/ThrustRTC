#include <Python.h>
#include "TRTCContext.h"
#include "replace.h"

static PyObject* n_replace(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DeviceViewable* old_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DeviceViewable* new_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	if (TRTC_Replace(*vec, *old_value, *new_value))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_replace_if(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DeviceViewable* new_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	if (TRTC_Replace_If(*vec, *pred, *new_value))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_replace_copy(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DeviceViewable* old_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DeviceViewable* new_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	if (TRTC_Replace_Copy(*vec_in, *vec_out, *old_value, *new_value))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_replace_copy_if(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DeviceViewable* new_value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	if(TRTC_Replace_Copy_If(*vec_in, *vec_out, *pred, *new_value))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
