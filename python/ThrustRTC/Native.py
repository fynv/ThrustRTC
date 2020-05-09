import os
import sys
import site
from .cffi import ffi

if os.name == 'nt':
    fn_thrustrtc = 'PyThrustRTC.dll'
elif os.name == "posix":
    fn_thrustrtc = 'libPyThrustRTC.so'

path_thrustrtc = sys.prefix+"/Fei/"+fn_thrustrtc
if not os.path.isfile(path_thrustrtc):
    path_thrustrtc = site.USER_BASE+"/Fei/"+fn_thrustrtc
    if not os.path.isfile(path_thrustrtc):
        path_thrustrtc = os.path.dirname(__file__)+"/../"+fn_thrustrtc

native = ffi.dlopen(path_thrustrtc)

