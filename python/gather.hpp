#include <Python.h>
#include "TRTCContext.h"
#include "gather.h"

static PyObject* n_gather(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_map = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	if (TRTC_Gather(*vec_map, *vec_in, *vec_out))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}

static PyObject* n_gather_if(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_map = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_stencil = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	PyObject *py_pred = PyTuple_GetItem(args, 4);
	Functor* pred = nullptr;
	if (py_pred != Py_None)
		pred = (Functor*)PyLong_AsVoidPtr(py_pred);

	if (pred == nullptr)
	{
		if (TRTC_Gather_If(*vec_map, *vec_stencil, *vec_in, *vec_out))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Gather_If(*vec_map, *vec_stencil, *vec_in, *vec_out, *pred))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}

