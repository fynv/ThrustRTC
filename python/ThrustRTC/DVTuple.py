from .Native import ffi, native
from .DeviceViewable import DeviceViewable
from .utils import *

class DVTuple(DeviceViewable):
	def __init__(self, elem_map):
		self.m_elem_map = elem_map
		param_names = [param_name for param_name, elem in elem_map.items()]
		o_param_names = StrArray(param_names)
		elems = [elem for param_name, elem in elem_map.items()]
		o_elems = ObjArray(elems)
		self.m_cptr = native.n_dvtuple_create(o_elems.m_cptr, o_param_names.m_cptr)

