from . import PyThrustRTC as native
from .Context import *
from .DVVector import *
from .DeviceViewable import *

def Sequence(ctx, vec, value_init=None, value_step=None, begin =0, end =-1):
	cptr_init = None
	if value_init!=None:
		cptr_init = value_init.m_cptr
	cptr_step = None
	if value_step!=None:
		cptr_step = value_step.m_cptr
	native.n_sequence(ctx.m_cptr, vec.m_cptr, cptr_init, cptr_step, begin, end)
