#include <Python.h>
#include "TRTCContext.h"
#include "reduce.h"
#include "viewbuf_to_python.hpp"

static PyObject* n_reduce(PyObject* self, PyObject* args)
{
	TRTCContext* ctx = (TRTCContext*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	DVVectorLike* vec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 2));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	ViewBuf ret;
	if (TRTC_Reduce(*ctx, *vec, ret, begin, end))
		return PyValue_FromViewBuf(ret, vec->name_elem_cls().c_str());
	Py_RETURN_NONE;
}
