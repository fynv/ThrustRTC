from . import PyThrustRTC as native
from .DeviceViewable import DeviceViewable

class DVTuple(DeviceViewable):
	def __init__(self, ctx, elem_map):
		self.m_elem_map = elem_map
		self.m_cptr = native.n_dvtuple_create(
			ctx.m_cptr,  
			[ (param_name, elem.m_cptr) for param_name, elem in elem_map.items()])

