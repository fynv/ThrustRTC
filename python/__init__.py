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
from .DVVector import *
from .Fill import *
from .Functor import *
from .Replace import *
from .ForEach import *
from .AdjacentDifference import *
from .Sequence import *
from .Tabulate import *
