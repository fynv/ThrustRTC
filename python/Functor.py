from . import PyThrustRTC as native

class Functor:
	def __init__(self, arg_map, functor_params, functor_ret, code_body):
		self.arg_map = [(param_name, arg.m_cptr) for param_name, arg in arg_map.items()]
		self.functor_params = functor_params
		self.functor_ret = functor_ret
		self.code_body = code_body

	def generate_code (self, type_ret, args):
		return native.n_functor_generate_code(self, type_ret, args)
