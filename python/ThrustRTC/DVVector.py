from . import PyThrustRTC as native
import numpy as np
from .DeviceViewable import DeviceViewable

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

def device_vector_from_list(ctx, lst, elem_cls):
	devive_viewables = [ native.n_dv_create_basic(elem_cls, item) for item in lst]
	dvec = DVVector(native.n_dvvector_from_dvs(ctx.m_cptr, devive_viewables))
	for dv in devive_viewables:
		native.n_dv_destroy(dv)
	return dvec
