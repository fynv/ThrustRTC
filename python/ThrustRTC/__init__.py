from . import PyThrustRTC as native
import os

def set_libnvrtc_path(path):
	native.n_set_libnvrtc_path(path)

def set_ptx_cache(path):
    if not os.path.exists(path):
        os.makedirs(path)
    native.n_set_ptx_cache(path)

from .Context import *
from .DeviceViewable import *
from .DVVector import device_vector, device_vector_from_numpy, device_vector_from_dvs, device_vector_from_list
from .Functor import *
from .Transformations import *
from .Copying import *
