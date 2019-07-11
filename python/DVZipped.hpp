#include <Python.h>
#include "TRTCContext.h"
#include "fake_vectors/DVZipped.h"

static PyObject* n_dvzipped_create(PyObject* self, PyObject* args)
{
	PyObject* pyvecs = PyTuple_GetItem(args, 0);
	ssize_t num_vecs = PyList_Size(pyvecs);
	std::vector<DVVectorLike*> vecs(num_vecs);
	for (ssize_t i = 0; i < num_vecs; i++)
		vecs[i] = (DVVectorLike*)PyLong_AsVoidPtr(PyList_GetItem(pyvecs, i));

	PyObject* py_elem_names = PyTuple_GetItem(args, 1);
	ssize_t num_elems = PyList_Size(py_elem_names);
	if (num_elems!= num_vecs)
	{
		PyErr_Format(PyExc_ValueError, "Number of vectors %d mismatch with number of element names %d.", num_vecs, num_elems);
		Py_RETURN_NONE;
	}
	std::vector<const char*> elem_names(num_elems);
	for (ssize_t i = 0; i < num_elems; i++)
		elem_names[i] = PyUnicode_AsUTF8(PyList_GetItem(py_elem_names, i));
	return PyLong_FromVoidPtr(new DVZipped(vecs, elem_names));

}

