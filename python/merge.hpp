#include <Python.h>
#include "TRTCContext.h"
#include "merge.h"

static PyObject* n_merge(PyObject* self, PyObject* args)
{
	DVVectorLike* vec1 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec2 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* vec_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	PyObject* py_comp = PyTuple_GetItem(args, 3);
	Functor* comp = nullptr;
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);

	if (comp == nullptr)
	{
		if (TRTC_Merge(*vec1, *vec2, *vec_out))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Merge(*vec1, *vec2, *vec_out, *comp))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
}

static PyObject* n_merge_by_key(PyObject* self, PyObject* args)
{
	
	DVVectorLike* keys1 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* keys2 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	DVVectorLike* value1 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVectorLike* value2 = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 3));
	DVVectorLike* keys_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 4));
	DVVectorLike* value_out = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 5));
	PyObject* py_comp = PyTuple_GetItem(args, 6);
	Functor* comp = nullptr;
	if (py_comp != Py_None)
		comp = (Functor*)PyLong_AsVoidPtr(py_comp);

	if (comp == nullptr)
	{
		if (TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out, *comp))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}

}
