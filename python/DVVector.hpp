#include <Python.h>
#include "TRTCContext.h"
#include "DVVector.h"

static PyObject* n_dvvectorlike_name_elem_cls(PyObject* self, PyObject* args)
{
	DVVectorLike* dvvec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyUnicode_FromString(dvvec->name_elem_cls().c_str());
}

static PyObject* n_dvvectorlike_size(PyObject* self, PyObject* args)
{
	DVVectorLike* dvvec = (DVVectorLike*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLongLong((long long)dvvec->size());
}

static PyObject* n_dvvector_create(PyObject* self, PyObject* args)
{
	const char* elem_cls = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	size_t size = (size_t)PyLong_AsUnsignedLongLong(PyTuple_GetItem(args, 1));
	PyObject* py_data = PyTuple_GetItem(args, 2);
	DVVector* ret = nullptr;
	if (py_data == Py_None)
		ret = new DVVector(elem_cls, size);
	else
	{
		void* data = PyLong_AsVoidPtr(py_data);
		ret = new DVVector(elem_cls, size, data);
	}
	return PyLong_FromVoidPtr(ret);
}

static PyObject* n_dvvector_to_host(PyObject* self, PyObject* args)
{
	DVVector* dvvec = (DVVector*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	void* ptr = PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	size_t begin = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 2));
	size_t end = (size_t)PyLong_AsLong(PyTuple_GetItem(args, 3));
	dvvec->to_host(ptr, begin, end);
	return PyLong_FromLong(0);
}

static PyObject* n_dvvector_from_dvs(PyObject* self, PyObject* args)
{
	PyObject* lst = PyTuple_GetItem(args, 0);
	ssize_t num_items = PyList_Size(lst);
	if (num_items < 1) Py_RETURN_NONE;
	PyObject* py_viewable = PyList_GetItem(lst, 0);
	DeviceViewable* dv = (DeviceViewable*)PyLong_AsVoidPtr(py_viewable);
	std::string elem_cls = dv->name_view_cls();
	for (ssize_t i = 1; i < num_items; i++)
	{
		py_viewable = PyList_GetItem(lst, i);
		dv = (DeviceViewable*)PyLong_AsVoidPtr(py_viewable);
		if (elem_cls != dv->name_view_cls())
			Py_RETURN_NONE;
	}
	size_t elem_size = TRTC_Size_Of(elem_cls.c_str());
	std::vector<char> buf(elem_size*num_items);
	for (ssize_t i = 0; i < num_items; i++)
	{
		py_viewable = PyList_GetItem(lst, i);
		dv = (DeviceViewable*)PyLong_AsVoidPtr(py_viewable);
		memcpy(buf.data() + elem_size*i, dv->view().data(), elem_size);
	}
	DVVector* ret = new DVVector(elem_cls.data(), num_items, buf.data());
	return PyLong_FromVoidPtr(ret);
}

static PyObject* n_dvvectoradaptor_create(PyObject* self, PyObject* args)
{
	const char* elem_cls = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	size_t size = (size_t)PyLong_AsUnsignedLongLong(PyTuple_GetItem(args, 1));
	void* data = PyLong_AsVoidPtr(PyTuple_GetItem(args, 2));
	DVVectorAdaptor* ret = new DVVectorAdaptor(elem_cls, size, data);
	return PyLong_FromVoidPtr(ret);
}