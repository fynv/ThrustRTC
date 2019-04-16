from . import PyThrustRTC as native
from .Context import *
from .DVVector import *
from .DeviceViewable import *
from .Functor import *

def Tabulate(ctx, vec, op, begin =0, end =-1):
	native.n_tabulate(ctx.m_cptr, vec.m_cptr, op, begin, end)
