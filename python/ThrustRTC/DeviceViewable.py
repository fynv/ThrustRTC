from . import PyThrustRTC as native

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