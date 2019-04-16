from . import PyThrustRTC as native
from .Context import *
from .DVVector import *
from .DeviceViewable import *
from .Functor import *

def Adjacent_Difference(ctx, vec_in, vec_out, binary_op=None, begin_in = 0, end_in = -1, begin_out = 0):
	native.n_adjacent_difference(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, binary_op, begin_in, end_in, begin_out)

