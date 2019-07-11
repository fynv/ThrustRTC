#include <Python.h>
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "viewbuf_to_python.hpp"

static PyObject* n_dv_name_view_cls(PyObject* self, PyObject* args)
{
	DeviceViewable* dv = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyUnicode_FromString(dv->name_view_cls().c_str());
}

static PyObject* n_dv_destroy(PyObject* self, PyObject* args)
{
	DeviceViewable* dv = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete dv;
	return PyLong_FromLong(0);
}

static PyObject* n_dv_create_basic(PyObject* self, PyObject* args)
{
	std::string type = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	PyObject* value = PyTuple_GetItem(args, 1);
	DeviceViewable* ret = nullptr;
	if (type == "int8_t")
		ret = new DVInt8((int8_t)PyLong_AsLong(value));
	else if (type == "uint8_t")
		ret = new DVUInt8((uint8_t)PyLong_AsLong(value));
	else if (type == "int16_t")
		ret = new DVInt16((int16_t)PyLong_AsLong(value));
	else if (type == "uint16_t")
		ret = new DVUInt16((uint16_t)PyLong_AsUnsignedLong(value));
	else if (type == "int32_t")
		ret = new DVInt32((int32_t)PyLong_AsLong(value));
	else if (type == "uint32_t")
		ret = new DVUInt32((uint32_t)PyLong_AsUnsignedLong(value));
	else if (type == "int64_t")
		ret = new DVInt64((int64_t)PyLong_AsLong(value));
	else if (type == "uint64_t")
		ret = new DVUInt64((uint64_t)PyLong_AsUnsignedLong(value));
	else if (type == "float")
		ret = new DVFloat((float)PyFloat_AsDouble(value));
	else if (type == "double")
		ret = new DVDouble((double)PyFloat_AsDouble(value));
	else if (type == "bool")
		ret = new DVBool(PyObject_IsTrue(value) != 0);
	return PyLong_FromVoidPtr(ret);
}

static PyObject* n_dv_value(PyObject* self, PyObject* args)
{
	DeviceViewable* dv = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyValue_FromViewBuf(dv->view(), dv->name_view_cls().c_str());
}

