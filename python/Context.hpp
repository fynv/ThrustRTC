#include <Python.h>
#include "TRTCContext.h"

static PyObject* n_set_verbose(PyObject* self, PyObject* args)
{
	bool verbose = PyObject_IsTrue(PyTuple_GetItem(args, 0)) != 0;
	TRTC_Set_Verbose(verbose);
	return PyLong_FromLong(0);
}

static PyObject* n_add_include_dir(PyObject* self, PyObject* args)
{
	const char* dir = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	TRTC_Add_Include_Dir(dir);
	return PyLong_FromLong(0);
}

static PyObject* n_add_built_in_header(PyObject* self, PyObject* args)
{
	const char* filename = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	const char* filecontent = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	TRTC_Add_Built_In_Header(filename, filecontent);
	return PyLong_FromLong(0);
}

static PyObject* n_add_inlcude_filename(PyObject* self, PyObject* args)
{
	const char* fn = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	TRTC_Add_Inlcude_Filename(fn);
	return PyLong_FromLong(0);
}

static PyObject* n_add_code_block(PyObject* self, PyObject* args)
{
	const char* line = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	TRTC_Add_Code_Block(line);
	return PyLong_FromLong(0);
}

static PyObject* n_add_constant_object(PyObject* self, PyObject* args)
{
	const char* name = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
	DeviceViewable* dv = (DeviceViewable*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	TRTC_Add_Constant_Object(name, *dv);
	return PyLong_FromLong(0);
}

static PyObject* n_wait(PyObject* self, PyObject* args)
{
	TRTC_Wait();
	return PyLong_FromLong(0);
}

static PyObject* n_kernel_create(PyObject* self, PyObject* args)
{
	PyObject* pyParamList = PyTuple_GetItem(args, 0);
	ssize_t num_params = PyList_Size(pyParamList);
	std::vector<const char*> params(num_params);
	for (ssize_t i = 0; i < num_params; i++)
		params[i] = PyUnicode_AsUTF8(PyList_GetItem(pyParamList, i));		
	const char* body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	TRTC_Kernel* cptr = new TRTC_Kernel(params, body);

	return PyLong_FromVoidPtr(cptr);
}

static PyObject* n_kernel_destroy(PyObject* self, PyObject* args)
{
	TRTC_Kernel* cptr = (TRTC_Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete cptr;
	return PyLong_FromLong(0);
}

static PyObject* n_kernel_num_params(PyObject* self, PyObject* args)
{
	TRTC_Kernel* cptr = (TRTC_Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLong((long)cptr->num_params());
}

static PyObject* n_kernel_calc_optimal_block_size(PyObject* self, PyObject* args)
{
	TRTC_Kernel* cptr = (TRTC_Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t num_params = cptr->num_params();

	PyObject* arg2 = PyTuple_GetItem(args, 1);
	std::vector<const DeviceViewable*> params;
	ssize_t size = PyList_Size(arg2);
	if ((ssize_t)num_params != size)
	{
		PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, size);
		Py_RETURN_NONE;
	}
	params.resize(size);
	for (ssize_t i = 0; i < size; i++)
		params[i] = (DeviceViewable*)PyLong_AsVoidPtr(PyList_GetItem(arg2, i));
	
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 2));
	int sizeBlock;
	if (cptr->calc_optimal_block_size(params.data(), sizeBlock, sharedMemBytes))
		return PyLong_FromLong((long)sizeBlock);
	else
		Py_RETURN_NONE;
}

static PyObject* n_kernel_calc_number_blocks(PyObject* self, PyObject* args)
{
	TRTC_Kernel* cptr = (TRTC_Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t num_params = cptr->num_params();

	PyObject* arg2 = PyTuple_GetItem(args, 1);
	std::vector<const DeviceViewable*> params;
	ssize_t size = PyList_Size(arg2);
	if ((ssize_t)num_params != size)
	{
		PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, size);
		Py_RETURN_NONE;
	}
	params.resize(size);
	for (ssize_t i = 0; i < size; i++)
		params[i] = (DeviceViewable*)PyLong_AsVoidPtr(PyList_GetItem(arg2, i));

	int sizeBlock = (int)PyLong_AsLongLong(PyTuple_GetItem(args, 2));
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 3));
	int numBlocks;
	if (cptr->calc_number_blocks(params.data(), sizeBlock, numBlocks, sharedMemBytes))
		return PyLong_FromLong((long)numBlocks);
	else
		Py_RETURN_NONE;
}

static PyObject* n_kernel_launch(PyObject* self, PyObject* args)
{
	TRTC_Kernel* cptr = (TRTC_Kernel*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t num_params = cptr->num_params();

	dim_type gridDim;
	PyObject* arg1 = PyTuple_GetItem(args, 1);
	if (PyObject_TypeCheck(arg1, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg1);
		gridDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 0)) : 1;
		gridDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 1)) : 1;
		gridDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg1, 2)) : 1;
	}
	else
	{
		gridDim.x = (unsigned)PyLong_AsUnsignedLong(arg1);
		gridDim.y = 1;
		gridDim.z = 1;
	}

	dim_type blockDim;
	PyObject* arg2 = PyTuple_GetItem(args, 2);
	if (PyObject_TypeCheck(arg2, &PyTuple_Type))
	{
		ssize_t size = PyTuple_Size(arg2);
		blockDim.x = size > 0 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 0)) : 1;
		blockDim.y = size > 1 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 1)) : 1;
		blockDim.z = size > 2 ? (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(arg2, 2)) : 1;
	}
	else
	{
		blockDim.x = (unsigned)PyLong_AsUnsignedLong(arg2);
		blockDim.y = 1;
		blockDim.z = 1;
	}

	PyObject* arg3 = PyTuple_GetItem(args, 3);
	std::vector<const DeviceViewable*> params;
	ssize_t size = PyList_Size(arg3);
	if ((ssize_t)num_params != size)
	{
		PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, size);
		Py_RETURN_NONE;
	}
	params.resize(size);
	for (ssize_t i = 0; i < size; i++)
		params[i] = (DeviceViewable*)PyLong_AsVoidPtr(PyList_GetItem(arg3, i));
	
	unsigned sharedMemBytes = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 4));
	if(cptr->launch(gridDim, blockDim, params.data(), sharedMemBytes))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}


static PyObject* n_for_create(PyObject* self, PyObject* args)
{
	PyObject* pyParamList = PyTuple_GetItem(args, 0);
	ssize_t num_params = PyList_Size(pyParamList);
	std::vector<const char*> params(num_params);
	for (ssize_t i = 0; i < num_params; i++)
		params[i] = PyUnicode_AsUTF8(PyList_GetItem(pyParamList, i));
	const char* name_iter = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	const char* body = PyUnicode_AsUTF8(PyTuple_GetItem(args, 2));
	TRTC_For* cptr = new TRTC_For(params, name_iter, body);

	return PyLong_FromVoidPtr(cptr);
}

static PyObject* n_for_destroy(PyObject* self, PyObject* args)
{
	TRTC_For* cptr = (TRTC_For*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete cptr;
	return PyLong_FromLong(0);
}

static PyObject* n_for_num_params(PyObject* self, PyObject* args)
{
	TRTC_For* cptr = (TRTC_For*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLong((long)cptr->num_params());
}

static PyObject* n_for_launch(PyObject* self, PyObject* args)
{
	TRTC_For* cptr = (TRTC_For*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t num_params = cptr->num_params();
	size_t begin = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));
	size_t end = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 2));

	PyObject* arg3 = PyTuple_GetItem(args, 3);
	std::vector<const DeviceViewable*> params;
	ssize_t size = PyList_Size(arg3);
	if ((ssize_t)num_params != size)
	{
		PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, size);
		Py_RETURN_NONE;
	}
	params.resize(size);
	for (ssize_t i = 0; i < size; i++)
		params[i] = (DeviceViewable*)PyLong_AsVoidPtr(PyList_GetItem(arg3, i));

	if(cptr->launch(begin, end, params.data()))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}


static PyObject* n_for_launch_n(PyObject* self, PyObject* args)
{
	TRTC_For* cptr = (TRTC_For*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	size_t num_params = cptr->num_params();
	size_t n = (size_t)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));

	PyObject* arg3 = PyTuple_GetItem(args, 2);
	std::vector<const DeviceViewable*> params;
	ssize_t size = PyList_Size(arg3);
	if ((ssize_t)num_params != size)
	{
		PyErr_Format(PyExc_ValueError, "Wrong number of arguments received. %d required, %d received.", num_params, size);
		Py_RETURN_NONE;
	}
	params.resize(size);
	for (ssize_t i = 0; i < size; i++)
		params[i] = (DeviceViewable*)PyLong_AsVoidPtr(PyList_GetItem(arg3, i));
	if (cptr->launch_n(n, params.data()))
		return PyLong_FromLong(0);
	else
		Py_RETURN_NONE;
}
