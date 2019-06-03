from . import PyThrustRTC as native

def Merge(ctx, vec1, vec2, vec_out, comp = None, begin1 = 0, end1 = -1, begin2 = 0, end2 = -1, begin_out = 0):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	native.n_merge(ctx.m_cptr, vec1.m_cptr, vec2.m_cptr, vec_out.m_cptr, cptr_comp, begin1, end1, begin2, end2, begin_out)

def Merge_By_Key(ctx, keys1, keys2, value1, value2, keys_out, value_out, comp = None, begin_keys1 = 0, end_keys1 = -1, begin_keys2 = 0, end_keys2 = -1, begin_value1 = 0, begin_value2 = 0, begin_keys_out = 0, begin_value_out = 0):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	native.n_merge_by_key(ctx.m_cptr, keys1.m_cptr, keys2.m_cptr, value1.m_cptr, value2.m_cptr, keys_out.m_cptr, value_out.m_cptr, cptr_comp, begin_keys1, end_keys1, begin_keys2, end_keys2, begin_value1, begin_value2, begin_keys_out, begin_value_out)

