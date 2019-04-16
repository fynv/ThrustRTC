from . import PyThrustRTC as native
from .Context import *
from .DVVector import *
from .DeviceViewable import *

def Fill(ctx, vec, value, begin =0, end =-1):
	native.n_fill(ctx.m_cptr, vec.m_cptr, value.m_cptr, begin, end)
