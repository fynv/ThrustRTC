from . import PyThrustRTC as native

def Sort(ctx, vec, comp = None, begin = 0, end = -1):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	native.n_sort(ctx.m_cptr, vec.m_cptr, cptr_comp, begin, end)

def Sort_By_Key(ctx, keys, values, comp = None, begin_keys = 0, end_keys = -1, begin_values = 0):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	native.n_sort_by_key(ctx.m_cptr, keys.m_cptr, values.m_cptr, cptr_comp, begin_keys, end_keys, begin_values)
