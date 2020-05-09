from .Native import ffi, native

def Find(vec, value):
	res = native.n_find(vec.m_cptr, value.m_cptr)
	if res==-1:
		return None
	return res

def Find_If(vec, pred):
	res = native.n_find_if(vec.m_cptr, pred.m_cptr)
	if res==-1:
		return None
	return res

def Find_If_Not(vec, pred):
	res = native.n_find_if_not(vec.m_cptr, pred.m_cptr)
	if res==-1:
		return None
	return res

def Mismatch(vec1, vec2, pred=None):
	cptr_pred = ffi.NULL
	if pred!=None:
		cptr_pred = pred.m_cptr
	res = native.n_mismatch(vec1.m_cptr, vec2.m_cptr, cptr_pred)
	if res==-1:
		return None
	return res

def Lower_Bound(vec, value, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_lower_bound(vec.m_cptr, value.m_cptr, cptr_comp)

def Upper_Bound(vec, value, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_upper_bound(vec.m_cptr, value.m_cptr, cptr_comp)

def Binary_Search(vec, value, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	res = native.n_binary_search(vec.m_cptr, value.m_cptr, cptr_comp)
	if res <0:
		return None
	return res!=0

def Partition_Point(vec, pred):
	return native.n_partition_point(vec.m_cptr, pred.m_cptr)

def Is_Sorted_Until(vec, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_is_sorted_until(vec.m_cptr, cptr_comp)

def Lower_Bound_V(vec, values, result, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	res = native.n_lower_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, cptr_comp)
	if res == -1:
		return None
	return res

def Upper_Bound_V(vec, values, result, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	res = native.n_upper_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, cptr_comp)
	if res == -1:
		return None
	return res

def Binary_Search_V(vec, values, result, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	res = native.n_binary_search_v(vec.m_cptr, values.m_cptr, result.m_cptr, cptr_comp)
	if res == -1:
		return None
	return res

