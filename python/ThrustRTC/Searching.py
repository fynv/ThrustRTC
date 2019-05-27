from . import PyThrustRTC as native

def Find(ctx, vec, value, begin = 0, end = -1):
	return native.n_find(ctx.m_cptr, vec.m_cptr, value.m_cptr, begin, end)

def Find_If(ctx, vec, pred, begin = 0, end = -1):
	return native.n_find_if(ctx.m_cptr, vec.m_cptr, pred.m_cptr, begin, end)

def Find_If_Not(ctx, vec, pred, begin = 0, end = -1):
	return native.n_find_if_not(ctx.m_cptr, vec.m_cptr, pred.m_cptr, begin, end)

def Mismatch(ctx, vec1, vec2, pred=None, begin1 = 0, end1 = -1, begin2 = 0):
	cptr_pred = None
	if pred!=None:
		cptr_pred = pred.m_cptr
	return native.n_mismatch(ctx.m_cptr, vec1.m_cptr, vec2.m_cptr, cptr_pred, begin1, end1, begin2)
