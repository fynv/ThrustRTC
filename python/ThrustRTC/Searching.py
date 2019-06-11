from . import PyThrustRTC as native

def Find(vec, value, begin = 0, end = -1):
	return native.n_find(vec.m_cptr, value.m_cptr, begin, end)

def Find_If(vec, pred, begin = 0, end = -1):
	return native.n_find_if(vec.m_cptr, pred.m_cptr, begin, end)

def Find_If_Not(vec, pred, begin = 0, end = -1):
	return native.n_find_if_not(vec.m_cptr, pred.m_cptr, begin, end)

def Mismatch(vec1, vec2, pred=None, begin1 = 0, end1 = -1, begin2 = 0):
	cptr_pred = None
	if pred!=None:
		cptr_pred = pred.m_cptr
	return native.n_mismatch(vec1.m_cptr, vec2.m_cptr, cptr_pred, begin1, end1, begin2)

def Lower_Bound(vec, value, comp = None, begin = 0, end = -1):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_lower_bound(vec.m_cptr, value.m_cptr, cptr_comp, begin, end)

def Upper_Bound(vec, value, comp = None, begin = 0, end = -1):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_upper_bound(vec.m_cptr, value.m_cptr, cptr_comp, begin, end)

def Binary_Search(vec, value, comp = None, begin = 0, end = -1):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_binary_search(vec.m_cptr, value.m_cptr, cptr_comp, begin, end)

def Partition_Point(vec, pred, begin = 0, end = -1):
	return native.n_partition_point(vec.m_cptr, pred.m_cptr, begin, end)

def Is_Sorted_Until(vec, comp = None, begin = 0, end = -1):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_is_sorted_until(vec.m_cptr, cptr_comp, begin, end)

def Lower_Bound_V(vec, values, result, comp = None, begin = 0, end = -1, begin_values = 0, end_values = -1, begin_result =0):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_lower_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, cptr_comp, begin, end, begin_values, end_values, begin_result)

def Upper_Bound_V(vec, values, result, comp = None, begin = 0, end = -1, begin_values = 0, end_values = -1, begin_result =0):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_upper_bound_v(vec.m_cptr, values.m_cptr, result.m_cptr, cptr_comp, begin, end, begin_values, end_values, begin_result)

def Binary_Search_V(vec, values, result, comp = None, begin = 0, end = -1, begin_values = 0, end_values = -1, begin_result =0):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_binary_search_v(vec.m_cptr, values.m_cptr, result.m_cptr, cptr_comp, begin, end, begin_values, end_values, begin_result)

