from . import PyThrustRTC as native
from .Context import *
from .DVVector import *
from .DeviceViewable import *
from .Functor import *

def For_Each(ctx, vec, f, begin =0, end =-1):
	native.n_for_each(ctx.m_cptr, vec.m_cptr, f, begin, end)
