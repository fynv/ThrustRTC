from .Native import ffi, native, check_cptr
from .DeviceViewable import DeviceViewable
from .utils import *

class Functor(DeviceViewable):
	def __init__(self, arg_map, functor_params, code_body):
		self.m_arg_map = arg_map
		param_names = [param_name for param_name, elem in arg_map.items()]
		o_param_names = StrArray(param_names)
		elems = [elem for param_name, elem in arg_map.items()]
		o_elems = ObjArray(elems)
		o_functor_params =  StrArray(functor_params)

		self.m_cptr = check_cptr(native.n_functor_create(			 
			o_elems.m_cptr, o_param_names.m_cptr,
			o_functor_params.m_cptr,
			code_body.encode('utf-8')))

class Identity(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"Identity"))

class Maximum(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"Maximum"))

class Minimum(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"Minimum"))


class EqualTo(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"EqualTo"))

class NotEqualTo(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"NotEqualTo"))

class Greater(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"Greater"))

class Less(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"Less"))

class GreaterEqual(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"GreaterEqual"))

class LessEqual(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"LessEqual"))

class Plus(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"Plus"))

class Minus(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"Minus"))

class Multiplies(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"Multiplies"))

class Divides(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"Divides"))

class Modulus(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"Modulus"))

class Negate(DeviceViewable):
	def __init__(self):
		self.m_cptr = check_cptr(native.n_built_in_functor_create(b"Negate"))
