#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVRange.h"

static PyObject* n_dvrange_create(PyObject* self, PyObject* args)
{
	DVVectorLike* vec_value = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t begin = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));
	size_t end = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 2));

	DVVector* p_vec = dynamic_cast<DVVector*>(vec_value);
	if (p_vec)
	{
		DVVectorAdaptor* ret = new DVVectorAdaptor(*p_vec, begin, end);
		return PyLong_FromVoidPtr(ret);
	}

	DVVectorAdaptor* p_vec_adpt = dynamic_cast<DVVectorAdaptor*>(vec_value);
	if (p_vec_adpt)
	{
		DVVectorAdaptor* ret = new DVVectorAdaptor(*p_vec_adpt, begin, end);
		return PyLong_FromVoidPtr(ret);
	}

	DVRange* ret = new DVRange(*vec_value, begin, end);
	return PyLong_FromVoidPtr(ret);
}
