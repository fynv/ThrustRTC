from . import PyThrustRTC as native

def Count(vec, value):
	return native.n_count(vec.m_cptr, value.m_cptr)

def Count_If(vec, pred):
	return native.n_count_if(vec.m_cptr, pred.m_cptr)

def Reduce(vec, value_init=None, binary_op=None):
	cptr_init = None
	if value_init!=None:
		cptr_init = value_init.m_cptr
	cptr_binary_op = None
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	return native.n_reduce(vec.m_cptr, cptr_init, cptr_binary_op)

def Equal(vec1, vec2, binary_pred=None):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	return native.n_equal(vec1.m_cptr, vec2.m_cptr, cptr_binary_pred)

def Min_Element(vec, comp = None):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_min_element(vec.m_cptr, cptr_comp)

def Max_Element(vec, comp = None):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_max_element(vec.m_cptr, cptr_comp)

def MinMax_Element(vec, comp = None):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_minmax_element(vec.m_cptr, cptr_comp)

def Inner_Product(vec1, vec2, value_init, binary_op1 = None, binary_op2 = None):
	cptr_binary_op1 = None
	if binary_op1!=None:
		cptr_binary_op1 = binary_op1.m_cptr
	cptr_binary_op2 = None
	if binary_op2!=None:
		cptr_binary_op2 = binary_op2.m_cptr
	return native.n_inner_product(vec1.m_cptr, vec2.m_cptr, value_init.m_cptr, cptr_binary_op1, cptr_binary_op2,)

def Transform_Reduce(vec, unary_op, value_init, binary_op):
	cptr_binary_op = None
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	cptr_unary_op = None
	if unary_op!=None:
		cptr_unary_op = unary_op.m_cptr
	return native.n_transform_reduce(vec.m_cptr, cptr_unary_op, value_init.m_cptr, cptr_binary_op)

def All_Of(vec, pred):
	cptr_pred = None
	if pred!=None:
		cptr_pred = pred.m_cptr
	return native.n_all_of(vec.m_cptr, cptr_pred)

def Any_Of(vec, pred):
	cptr_pred = None
	if pred!=None:
		cptr_pred = pred.m_cptr
	return native.n_any_of(vec.m_cptr, cptr_pred)

def None_Of(vec, pred):
	cptr_pred = None
	if pred!=None:
		cptr_pred = pred.m_cptr
	return native.n_none_of(vec.m_cptr, cptr_pred)

def Reduce_By_Key(key_in, value_in, key_out, value_out, binary_pred = None, binary_op = None):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	cptr_binary_op = None
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	return native.n_reduce_by_key(key_in.m_cptr, value_in.m_cptr, key_out.m_cptr, value_out.m_cptr, cptr_binary_pred, cptr_binary_op)

def Is_Partitioned(vec, pred):
	return native.n_is_partitioned(vec.m_cptr, pred.m_cptr)

def Is_Sorted(vec, comp = None):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_is_sorted(vec.m_cptr, cptr_comp)

