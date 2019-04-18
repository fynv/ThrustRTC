from . import PyThrustRTC as native

def Tabulate(ctx, vec, op, begin =0, end =-1):
	native.n_tabulate(ctx.m_cptr, vec.m_cptr, op, begin, end)
