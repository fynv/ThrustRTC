from . import PyThrustRTC as native
from .DeviceViewable import DeviceViewable

class DVTuple(DeviceViewable):
	def __init__(self, elem_map):
		self.m_elem_map = elem_map
		self.m_cptr = native.n_dvtuple_create(
			 
			[ (param_name, elem.m_cptr) for param_name, elem in elem_map.items()])

