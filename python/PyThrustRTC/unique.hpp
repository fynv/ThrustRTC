#include <Python.h>
#include "TRTCContext.h"
#include "unique.h"

static PyObject* n_unique(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	Functor* binary_pred = nullptr;
	PyObject* py_binary_pred = PyTuple_GetItem(args, 2);
	if (py_binary_pred!= Py_None)
		binary_pred=(Functor*)PyLong_AsVoidPtr(py_binary_pred);
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	if (binary_pred == nullptr)
	{
		uint32_t res = TRTC_Unique(*ctx, *vec, begin, end);
		if (res != uint32_t(-1))
			return PyLong_FromUnsignedLong((unsigned long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		uint32_t res = TRTC_Unique(*ctx, *vec, *binary_pred, begin, end);
		if (res != uint32_t(-1))
			return PyLong_FromUnsignedLong((unsigned long)res);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_unique_copy(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* binary_pred = nullptr;
	PyObject* py_binary_pred = PyTuple_GetItem(args, 3);
	if (py_binary_pred != Py_None)
		binary_pred = (Functor*)PyLong_AsVoidPtr(py_binary_pred);
	size_t begin_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	if (binary_pred == nullptr)
	{
		uint32_t res = TRTC_Unique_Copy(*ctx, *vec_in, *vec_out, begin_in, end_in, begin_out);
		if (res != uint32_t(-1))
			return PyLong_FromUnsignedLong((unsigned long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		uint32_t res = TRTC_Unique_Copy(*ctx, *vec_in, *vec_out, *binary_pred, begin_in, end_in, begin_out);
		if (res != uint32_t(-1))
			return PyLong_FromUnsignedLong((unsigned long)res);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_unique_by_key(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* keys = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* values = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	Functor* binary_pred = nullptr;
	PyObject* py_binary_pred = PyTuple_GetItem(args, 3);
	if (py_binary_pred != Py_None)
		binary_pred = (Functor*)PyLong_AsVoidPtr(py_binary_pred);
	size_t begin_key = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end_key = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t begin_value = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	if (binary_pred == nullptr)
	{
		uint32_t res = TRTC_Unique_By_Key(*ctx, *keys, *values, begin_key, end_key, begin_value);
		if (res != uint32_t(-1))
			return PyLong_FromUnsignedLong((unsigned long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		uint32_t res = TRTC_Unique_By_Key(*ctx, *keys, *values, *binary_pred, begin_key, end_key, begin_value);
		if (res != uint32_t(-1))
			return PyLong_FromUnsignedLong((unsigned long)res);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_unique_by_key_copy(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* keys_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* values_in = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVectorLike* keys_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	DVVectorLike* values_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 4));
	Functor* binary_pred = nullptr;
	PyObject* py_binary_pred = PyTuple_GetItem(args, 5);
	if (py_binary_pred != Py_None)
		binary_pred = (Functor*)PyLong_AsVoidPtr(py_binary_pred);
	size_t begin_key_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t end_key_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin_value_in = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));
	size_t begin_key_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 9));
	size_t begin_value_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 10));
	if (binary_pred == nullptr)
	{
		uint32_t res = TRTC_Unique_By_Key_Copy(*ctx, *keys_in, *values_in, *keys_out, *values_out, begin_key_in, end_key_in, begin_value_in, begin_key_out, begin_value_out);
		if (res != uint32_t(-1))
			return PyLong_FromUnsignedLong((unsigned long)res);
		else
			Py_RETURN_NONE;
	}
	else
	{
		uint32_t res = TRTC_Unique_By_Key_Copy(*ctx, *keys_in, *values_in, *keys_out, *values_out, *binary_pred, begin_key_in, end_key_in, begin_value_in, begin_key_out, begin_value_out);
		if (res != uint32_t(-1))
			return PyLong_FromUnsignedLong((unsigned long)res);
		else
			Py_RETURN_NONE;
	}
}

