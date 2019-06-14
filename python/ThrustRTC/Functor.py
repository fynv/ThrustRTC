from . import PyThrustRTC as native
from .DeviceViewable import DeviceViewable

class Functor(DeviceViewable):
	def __init__(self, arg_map, functor_params, code_body):
		self.m_arg_map = arg_map
		self.m_cptr = native.n_functor_create(			 
			[ (param_name, arg.m_cptr) for param_name, arg in arg_map.items()],
			functor_params,
			code_body)

class Identity(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("Identity")

class Maximum(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("Maximum")

class Minimum(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("Minimum")

class EqualTo(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("EqualTo")

class NotEqualTo(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("NotEqualTo")

class Greater(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("Greater")

class Less(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("Less")

class GreaterEqual(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("GreaterEqual")

class LessEqual(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("LessEqual")

class Plus(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("Plus")

class Minus(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("Minus")

class Multiplies(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("Multiplies")

class Divides(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("Divides")

class Modulus(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("Modulus")

class Negate(DeviceViewable):
	def __init__(self):
		self.m_cptr = native.n_built_in_functor_create("Negate")
