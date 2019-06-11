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
	size_t begin1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 4));
	size_t end1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 5));
	size_t begin2 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 6));
	size_t end2 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t begin_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));

	if (comp == nullptr)
	{
		if (TRTC_Merge(*vec1, *vec2, *vec_out, begin1, end1, begin2, end2, begin_out))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Merge(*vec1, *vec2, *vec_out, *comp, begin1, end1, begin2, end2, begin_out))
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
	size_t begin_keys1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 7));
	size_t end_keys1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 8));
	size_t begin_keys2 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 9));
	size_t end_keys2 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 10));
	size_t begin_value1 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 11));
	size_t begin_value2 = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 12));
	size_t begin_keys_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 13));
	size_t begin_value_out = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 14));

	if (comp == nullptr)
	{
		if (TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out, begin_keys1, end_keys1, begin_keys2, end_keys2, begin_value1, begin_value2, begin_keys_out, begin_value_out))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}
	else
	{
		if (TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out, *comp, begin_keys1, end_keys1, begin_keys2, end_keys2, begin_value1, begin_value2, begin_keys_out, begin_value_out))
			return PyLong_FromLong(0);
		else
			Py_RETURN_NONE;
	}

}
