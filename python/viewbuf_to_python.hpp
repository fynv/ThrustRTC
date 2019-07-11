#ifndef _viewbuf_to_python_hpp_
#define _viewbuf_to_python_hpp_

#include <Python.h>
#include "DeviceViewable.h"

PyObject* PyValue_FromViewBuf(const ViewBuf& buf, const char* type)
{
	std::string s_type = type;
	if (s_type == "int8_t")
		return PyLong_FromLong((long)(*(int8_t*)buf.data()));
	else if (s_type == "uint8_t")
		return PyLong_FromUnsignedLong((unsigned long)(*(uint8_t*)buf.data()));
	else if (s_type == "int16_t")
		return PyLong_FromLong((long)(*(int16_t*)buf.data()));
	else if (s_type == "uint16_t")
		return PyLong_FromUnsignedLong((unsigned long)(*(uint16_t*)buf.data()));
	else if (s_type == "int32_t")
		return PyLong_FromLong((long)(*(int32_t*)buf.data()));
	else if (s_type == "uint32_t")
		return PyLong_FromUnsignedLong((unsigned long)(*(uint32_t*)buf.data()));
	else if (s_type == "int64_t")
		return PyLong_FromLongLong((long long)(*(int64_t*)buf.data()));
	else if (s_type == "uint64_t")
		return PyLong_FromUnsignedLongLong((unsigned long long)(*(uint64_t*)buf.data()));
	else if (s_type == "float")
		return PyFloat_FromDouble((double)(*(float*)buf.data()));
	else if (s_type == "double")
		return PyFloat_FromDouble(*(double*)buf.data());
	else if (s_type == "bool")
		return PyBool_FromLong(*(bool*)buf.data()?1:0);
	
	char str[64];
	sprintf(str, "[Device-viewable object, type: %s]", type);
	return PyUnicode_FromString(str);
}

#endif
