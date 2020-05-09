from .Native import ffi, native

def Sort(vec, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	native.n_sort(vec.m_cptr, cptr_comp)

def Sort_By_Key(keys, values, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	native.n_sort_by_key(keys.m_cptr, values.m_cptr, cptr_comp)
