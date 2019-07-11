#include <Python.h>
#include "TRTCContext.h"
#include "logical.h"

static PyObject* n_all_of(PyObject* self, PyObject* args)
{	
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	bool res;
	if (TRTC_All_Of(*vec, *pred, res))
		return PyBool_FromLong(res ? 1 : 0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_any_of(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	bool res;
	if (TRTC_Any_Of(*vec, *pred, res))
		return PyBool_FromLong(res ? 1 : 0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_none_of(PyObject* self, PyObject* args)
{	
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	bool res;
	if (TRTC_None_Of(*vec, *pred, res))
		return PyBool_FromLong(res ? 1 : 0);
	else
		Py_RETURN_NONE;
}
