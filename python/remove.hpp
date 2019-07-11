#include <Python.h>
#include "TRTCContext.h"
#include "remove.h"

static PyObject* n_remove(PyObject* self, PyObject* args)
{
	DVVectorLike* vec= (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	uint32_t res = TRTC_Remove(*vec, *value);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_remove_copy(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DeviceViewable* value = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	uint32_t res = TRTC_Remove_Copy(*vec_in, *vec_out, *value);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_remove_if(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	uint32_t res = TRTC_Remove_If(*vec, *pred);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_remove_copy_if(PyObject* self, PyObject* args)
{	
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	uint32_t res = TRTC_Remove_Copy_If(*vec_in, *vec_out, *pred);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_remove_if_stencil(PyObject* self, PyObject* args)
{
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* stencil = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	uint32_t res = TRTC_Remove_If_Stencil(*vec, *stencil, *pred);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}

static PyObject* n_remove_copy_if_stencil(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* stencil = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* pred = (Functor*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	uint32_t res = TRTC_Remove_Copy_If_Stencil(*vec_in, *stencil, *vec_out, *pred);
	if (res != uint32_t(-1))
		return PyLong_FromUnsignedLong((unsigned long)res);
	else
		Py_RETURN_NONE;
}


