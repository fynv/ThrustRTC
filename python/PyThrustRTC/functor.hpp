#include <Python.h>
#include "functor.h"

static PyObject* n_functor_create(PyObject* self, PyObject* args)
{
	PyObject* pyArgMap = PyTuple_GetItem(args, 0);
	ssize_t num_params = PyList_Size(pyArgMap);
	std::vector<AssignedParam> arg_map(num_params);
	for (ssize_t i = 0; i < num_params; i++)
	{
		PyObject* pyAssignedParam = PyList_GetItem(pyArgMap, i);
		arg_map[i].param_name = PyUnicode_AsUTF8(PyTuple_GetItem(pyAssignedParam, 0));
		arg_map[i].arg = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(pyAssignedParam, 1));
	}
	PyObject* py_functor_params = PyTuple_GetItem(args, 1);
	ssize_t num_functor_params = PyList_Size(py_functor_params);
	std::vector<const char*> functor_params(num_functor_params);
	for (ssize_t i = 0; i < num_functor_params; i++)
		functor_params[i] = PyUnicode_AsUTF8(PyList_GetItem(py_functor_params, i));

	PyObject* py_code_body = PyTuple_GetItem(args, 2);
	const char* code_body = PyUnicode_AsUTF8(py_code_body);
	return PyLong_FromVoidPtr(new Functor(arg_map, functor_params, code_body));
}

static PyObject* n_built_in_functor_create(PyObject* self, PyObject* args)
{
	const char* name_built_in_view_cls = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	return PyLong_FromVoidPtr(new Functor(name_built_in_view_cls));
}
