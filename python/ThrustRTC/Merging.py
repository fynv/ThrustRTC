from . import PyThrustRTC as native

def Merge(vec1, vec2, vec_out, comp = None):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	native.n_merge(vec1.m_cptr, vec2.m_cptr, vec_out.m_cptr, cptr_comp)

def Merge_By_Key(keys1, keys2, value1, value2, keys_out, value_out, comp = None):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	native.n_merge_by_key(keys1.m_cptr, keys2.m_cptr, value1.m_cptr, value2.m_cptr, keys_out.m_cptr, value_out.m_cptr, cptr_comp)

