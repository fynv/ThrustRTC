from . import PyThrustRTC as native
from .DeviceViewable import DeviceViewable

class Functor(DeviceViewable):
	def __init__(self, ctx, arg_map, functor_params, code_body):
		self.m_cptr = native.n_functor_create(
			ctx.m_cptr,  
			[ (param_name, arg.m_cptr) for param_name, arg in arg_map.items()],
			functor_params,
			code_body)

