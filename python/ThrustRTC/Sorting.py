from . import PyThrustRTC as native

def Sort(vec, comp = None, begin = 0, end = -1):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	native.n_sort(vec.m_cptr, cptr_comp, begin, end)

def Sort_By_Key(keys, values, comp = None, begin_keys = 0, end_keys = -1, begin_values = 0):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	native.n_sort_by_key(keys.m_cptr, values.m_cptr, cptr_comp, begin_keys, end_keys, begin_values)
