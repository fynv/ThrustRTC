from .Native import ffi, native, check_i, check_ull

def Find(vec, value):
	res = check_i(native.n_find(vec.m_cptr, value.m_cptr))
	if res>=vec.size():
		return None
	return res

def Find_If(vec, pred):
	res = check_i(native.n_find_if(vec.m_cptr, pred.m_cptr))
	if res>=vec.size():
		return None
	return res

def Find_If_Not(vec, pred):
	res = check_i(native.n_find_if_not(vec.m_cptr, pred.m_cptr))
	if res>=vec.size():
		return None
	return res

def Mismatch(vec1, vec2, pred=None):
	cptr_pred = ffi.NULL
	if pred!=None:
		cptr_pred = pred.m_cptr
	res = check_i(native.n_mismatch(vec1.m_cptr, vec2.m_cptr, cptr_pred))
	if res>=vec1.size() and res>=vec2.size():
		return None
	return res

def Lower_Bound(vec, value, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	return check_ull(native.n_lower_bound(vec.m_cptr, value.m_cptr, cptr_comp))

def Upper_Bound(vec, value, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	return check_ull(native.n_upper_bound(vec.m_cptr, value.m_cptr, cptr_comp))

def Binary_Search(vec, value, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	res = check_i(native.n_binary_search(vec.m_cptr, value.m_cptr, cptr_comp))
	if res <0:
		return None
	return res!=0

def Partition_Point(vec, pred):
	return check_i(native.n_partition_point(vec.m_cptr, pred.m_cptr))

def Is_Sorted_Until(vec, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	return check_ull(native.n_is_sorted_until(vec.m_cptr, cptr_comp))

def Lower_Bound_V(vec, values, result, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	res = check_i(native.n_lower_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, cptr_comp))
	if res == -1:
		return None
	return res

def Upper_Bound_V(vec, values, result, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	res = check_i(native.n_upper_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, cptr_comp))
	if res == -1:
		return None
	return res

def Binary_Search_V(vec, values, result, comp = None):
	cptr_comp = ffi.NULL
	if comp!=None:
		cptr_comp = comp.m_cptr
	res = check_i(native.n_binary_search_v(vec.m_cptr, values.m_cptr, result.m_cptr, cptr_comp))
	if res == -1:
		return None
	return res

