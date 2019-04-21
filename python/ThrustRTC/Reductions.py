from . import PyThrustRTC as native

def Count(ctx, vec, value, begin =0, end =-1):
	return native.n_count(ctx.m_cptr, vec.m_cptr, value.m_cptr, begin, end)

def Count_If(ctx, vec, pred, begin =0, end =-1):
	return native.n_count_if(ctx.m_cptr, vec.m_cptr, pred, begin, end)
