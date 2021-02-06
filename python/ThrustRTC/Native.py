import os
import sys
import site
from .cffi import ffi

if os.name == 'nt':
    fn_thrustrtc = 'PyThrustRTC.dll'
elif os.name == "posix":
    fn_thrustrtc = 'libPyThrustRTC.so'
    
path_thrustrtc = os.path.dirname(__file__)+"/"+fn_thrustrtc
native = ffi.dlopen(path_thrustrtc)

def check_i(ret_value_i):
	if ret_value_i == -1:
		raise SystemError("An internal error happend")
	elif ret_value_i == -2:
		raise ValueError("Wrong number of arguments.")
	elif ret_value_i == -100:
		raise ImportError('cannot import ThrustRTC')
	return ret_value_i

def check_u(ret_value_u):
	if ret_value_u == 0xFFFFFFFF:
		raise SystemError("An internal error happend")
	return ret_value_u	

def check_ull(ret_value_ull):
	if ret_value_ull == 0xFFFFFFFFFFFFFFFF:
		raise SystemError("An internal error happend")
	return ret_value_ull

def check_cptr(ret_value_cptr):
	if ret_value_cptr == ffi.NULL:
		raise SystemError("An internal error happend")
	return ret_value_cptr

		