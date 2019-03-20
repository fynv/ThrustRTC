import PyThrustRTC as native
import os
import numpy as np

def set_ptx_cache(path):
    if not os.path.exists(path):
        os.makedirs(path)
    native.n_set_ptx_cache(path)

class Context:
    def __init__(self):
    	self.m_cptr = native.n_context_create()

    def __del__(self):
    	native.n_context_destroy(self.m_cptr)

    def set_verbose(self, verbose=True):
        native.n_context_set_verbose(self.m_cptr, verbose)

    def add_include_dir(self, path):
        native.n_context_add_include_dir(self.m_cptr, path)

    def add_inlcude_filename(self, filename):
        native.n_context_add_inlcude_filename(self.m_cptr, filename)

    def add_preprocessor(self, line):
        native.n_context_add_preprocessor(self.m_cptr, line)

class DeviceViewable:
	def name_view_cls(self):
		return native.n_dv_name_view_cls(self.m_cptr)
	def __del__(self):
		native.n_dv_destroy(self.m_cptr)

class DVInt8(DeviceViewable):
	def __init__(self, value):
		self.m_cptr = native.n_dv_create_basic('int8_t', value)

class DVUInt8(DeviceViewable):
	def __init__(self, value):
		self.m_cptr = native.n_dv_create_basic('uint8_t', value)

class DVInt16(DeviceViewable):
	def __init__(self, value):
		self.m_cptr = native.n_dv_create_basic('int16_t', value)

class DVUInt16(DeviceViewable):
	def __init__(self, value):
		self.m_cptr = native.n_dv_create_basic('uint16_t', value)

class DVInt32(DeviceViewable):
	def __init__(self, value):
		self.m_cptr = native.n_dv_create_basic('int32_t', value)

class DVUInt32(DeviceViewable):
	def __init__(self, value):
		self.m_cptr = native.n_dv_create_basic('uint32_t', value)

class DVInt64(DeviceViewable):
	def __init__(self, value):
		self.m_cptr = native.n_dv_create_basic('int64_t', value)

class DVUInt64(DeviceViewable):
	def __init__(self, value):
		self.m_cptr = native.n_dv_create_basic('uint64_t', value)

class DVFloat(DeviceViewable):
	def __init__(self, value):
		self.m_cptr = native.n_dv_create_basic('float', value)

class DVDouble(DeviceViewable):
	def __init__(self, value):
		self.m_cptr = native.n_dv_create_basic('double', value)

class DVBool(DeviceViewable):
	def __init__(self, value):
		self.m_cptr = native.n_dv_create_basic('bool', value)

class DVVector(DeviceViewable):
	def __init__(self, cptr):
		self.m_cptr = cptr

	def name_elem_cls(self):
		return native.n_dvvector_name_elem_cls(self.m_cptr)

	def size(self):
		return native.n_dvvector_size(self.m_cptr)

	def to_host(self):
		elem_cls = self.name_elem_cls()
		if elem_cls=='int8_t':
			nptype = np.int8
		elif elem_cls=='uint8_t':
			nptype = np.uint8
		elif elem_cls=='int16_t':
			nptype = np.int16
		elif elem_cls=='uint16_t':
			nptype = np.uint16
		elif elem_cls=='int32_t':
			nptype = np.int32
		elif elem_cls=='uint32_t':
			nptype = np.uint32
		elif elem_cls=='int64_t':
			nptype = np.int64
		elif elem_cls=='uint64_t':
			nptype = np.uint64
		elif elem_cls=='float':
			nptype = np.float32
		elif elem_cls=='double':
			nptype = np.float64
		elif elem_cls=='bool':
			nptype = np.bool
		ret = np.empty(self.size(), dtype=nptype)
		native.n_dvvector_to_host(self.m_cptr, ret.__array_interface__['data'][0])
		return ret

def device_vector(ctx, elem_cls, size, ptr_host_data=None):
	return DVVector(native.n_dvvector_create(ctx.m_cptr, elem_cls, size, ptr_host_data))

def device_vector_from_numpy(ctx, nparr):
	if nparr.dtype == np.int8:
		elem_cls = 'int8_t'
	elif nparr.dtype == np.uint8:
		elem_cls = 'uint8_t'
	elif nparr.dtype == np.int16:
		elem_cls = 'int16_t'
	elif nparr.dtype == np.uint16:
		elem_cls = 'uint16_t'
	elif nparr.dtype == np.int32:
		elem_cls = 'int32_t'
	elif nparr.dtype == np.uint32:
		elem_cls = 'uint32_t'		
	elif nparr.dtype == np.int64:
		elem_cls = 'int64_t'
	elif nparr.dtype == np.uint64:
		elem_cls = 'uint64_t'	
	elif nparr.dtype == np.int64:
		elem_cls = 'int64_t'
	elif nparr.dtype == np.uint64:
		elem_cls = 'uint64_t'	
	elif nparr.dtype == np.float32:
		elem_cls = 'float'
	elif nparr.dtype == np.float64:
		elem_cls = 'double'
	elif nparr.dtype == np.bool:
		elem_cls = 'bool'
	size = len(nparr)
	ptr_host_data = nparr.__array_interface__['data'][0]
	return device_vector(ctx, elem_cls, size, ptr_host_data)

def device_vector_from_dvs(ctx, lst_dv):
	return DVVector(native.n_dvvector_from_dvs(ctx.m_cptr, [item.m_cptr for item in lst_dv]))

class Kernel:
	def __init__(self, ctx, param_descs, body):
		self.m_cptr = native.n_kernel_create(ctx.m_cptr, param_descs, body)

	def __del__(self):
		native.n_kernel_destroy(self.m_cptr)

	def num_params(self):
		return native.n_kernel_num_params(self.m_cptr)

	def launch(self, gridDim, blockDim, args, sharedMemBytes=0):
		native.n_kernel_launch(self.m_cptr, gridDim, blockDim, [item.m_cptr for item in args], sharedMemBytes)

class KernelTemplate:
	def __init__(self, ctx, template_params, param_descs, body):
		self.m_cptr = native.n_kernel_template_create(ctx.m_cptr, template_params, param_descs, body)

	def __del__(self):
		native.n_kernel_template_destroy(self.m_cptr)

	def num_template_params(self):
		return native.n_kernel_template_num_template_params(self.m_cptr)

	def num_params(self):
		return native.n_kernel_template_num_params(self.m_cptr)

	def launch_explict(self, gridDim, blockDim, template_args, args, sharedMemBytes=0):
		native.n_kernel_template_launch_explict(self.m_cptr, gridDim, blockDim, template_args, [item.m_cptr for item in args], sharedMemBytes)

	def launch(self, gridDim, blockDim, args, sharedMemBytes=0):
		native.n_kernel_template_launch(self.m_cptr, gridDim, blockDim, [item.m_cptr for item in args], sharedMemBytes)


