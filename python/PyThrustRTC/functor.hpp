#ifndef _functor_hpp_
#define _functor_hpp_

#include <Python.h>
#include "functor.h"

inline Functor PyFunctor_AsFunctor(PyObject* pyFunctor)
{
	Functor functor;
	PyObject* py_arg_map = PyObject_GetAttrString(pyFunctor, "arg_map");
	ssize_t num_params = PyList_Size(py_arg_map);
	functor.arg_map.resize((size_t)num_params);
	for (ssize_t i = 0; i < num_params; i++)
	{
		PyObject* pyAssignedParam = PyList_GetItem(py_arg_map, i);
		functor.arg_map[i].param_name = PyUnicode_AsUTF8(PyTuple_GetItem(pyAssignedParam, 0));
		functor.arg_map[i].arg = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(pyAssignedParam, 1));
	}

	PyObject* py_functor_params = PyObject_GetAttrString(pyFunctor, "functor_params");
	ssize_t num_functor_params = PyList_Size(py_functor_params);
	functor.functor_params.resize((size_t)num_functor_params);
	for (ssize_t i = 0; i < num_functor_params; i++)
		functor.functor_params[i] = PyUnicode_AsUTF8(PyList_GetItem(py_functor_params, i));
	
	PyObject* py_functor_ret = PyObject_GetAttrString(pyFunctor, "functor_ret");
	if (py_functor_ret == Py_None)
		functor.functor_ret = nullptr;
	else
		functor.functor_ret = PyUnicode_AsUTF8(py_functor_ret);

	PyObject* py_code_body = PyObject_GetAttrString(pyFunctor, "code_body");
	functor.code_body = PyUnicode_AsUTF8(py_code_body);

	return functor;
}


static PyObject* n_functor_generate_code(PyObject* self, PyObject* args)
{
	Functor functor = PyFunctor_AsFunctor(PyTuple_GetItem(args, 0));
	const char* type_ret = nullptr;
	PyObject* py_type_ret = PyTuple_GetItem(args, 1);
	if (py_type_ret != Py_None)
		type_ret = PyUnicode_AsUTF8(py_type_ret);

	size_t num_params = functor.functor_params.size();
	std::vector<const char*> lst_args(num_params);
	PyObject* py_list_args = PyTuple_GetItem(args, 2);
	
	ssize_t num_args = PyList_Size(py_list_args);
	if (num_args != (ssize_t)num_params)
	{
		PyErr_Format(PyExc_ValueError, "Wrong number of functor arguments received. %d required, %d received.", num_params, num_args);
		Py_RETURN_NONE;
	}

	for (size_t i = 0; i < num_params; i++)
		lst_args[i] = PyUnicode_AsUTF8(PyList_GetItem(py_list_args, i));

	return PyUnicode_FromString(functor.generate_code(type_ret, lst_args).c_str());
}

#endif