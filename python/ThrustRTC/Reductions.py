from . import PyThrustRTC as native

def Count(ctx, vec, value, begin = 0, end = -1):
	return native.n_count(ctx.m_cptr, vec.m_cptr, value.m_cptr, begin, end)

def Count_If(ctx, vec, pred, begin = 0, end = -1):
	return native.n_count_if(ctx.m_cptr, vec.m_cptr, pred.m_cptr, begin, end)

def Reduce(ctx, vec, value_init=None, binary_op=None, begin = 0, end = -1):
	cptr_init = None
	if value_init!=None:
		cptr_init = value_init.m_cptr
	cptr_binary_op = None
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	return native.n_reduce(ctx.m_cptr, vec.m_cptr, cptr_init, cptr_binary_op, begin, end)

def Equal(ctx, vec1, vec2, binary_pred=None, begin1 = 0, end1 = -1, begin2 = 0 ):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	return native.n_equal(ctx.m_cptr, vec1.m_cptr, vec2.m_cptr, cptr_binary_pred, begin1, end1, begin2)

def Min_Element(ctx, vec, comp = None, begin = 0, end = -1):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_min_element(ctx.m_cptr, vec.m_cptr, cptr_comp, begin, end)

def Max_Element(ctx, vec, comp = None, begin = 0, end = -1):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_max_element(ctx.m_cptr, vec.m_cptr, cptr_comp, begin, end)

def MinMax_Element(ctx, vec, comp = None, begin = 0, end = -1):
	cptr_comp = None
	if comp!=None:
		cptr_comp = comp.m_cptr
	return native.n_minmax_element(ctx.m_cptr, vec.m_cptr, cptr_comp, begin, end)

def Inner_Product(ctx, vec1, vec2, value_init, binary_op1 = None, binary_op2 = None, begin1 = 0, end1 = -1, begin2 = 0):
	cptr_binary_op1 = None
	if binary_op1!=None:
		cptr_binary_op1 = binary_op1.m_cptr
	cptr_binary_op2 = None
	if binary_op2!=None:
		cptr_binary_op2 = binary_op2.m_cptr
	return native.n_inner_product(ctx.m_cptr, vec1.m_cptr, vec2.m_cptr, value_init.m_cptr, cptr_binary_op1, cptr_binary_op2, begin1, end1, begin2)

def Transform_Reduce(ctx, vec, unary_op, value_init, binary_op, begin = 0, end = -1):
	cptr_binary_op = None
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	cptr_unary_op = None
	if unary_op!=None:
		cptr_unary_op = unary_op.m_cptr
	return native.n_transform_reduce(ctx.m_cptr, vec.m_cptr, cptr_unary_op, value_init.m_cptr, cptr_binary_op, begin, end)

def All_Of(ctx, vec, pred, begin = 0, end = -1):
	cptr_pred = None
	if pred!=None:
		cptr_pred = pred.m_cptr
	return native.n_all_of(ctx.m_cptr, vec.m_cptr, cptr_pred, begin, end)

def Any_Of(ctx, vec, pred, begin = 0, end = -1):
	cptr_pred = None
	if pred!=None:
		cptr_pred = pred.m_cptr
	return native.n_any_of(ctx.m_cptr, vec.m_cptr, cptr_pred, begin, end)

def None_Of(ctx, vec, pred, begin = 0, end = -1):
	cptr_pred = None
	if pred!=None:
		cptr_pred = pred.m_cptr
	return native.n_none_of(ctx.m_cptr, vec.m_cptr, cptr_pred, begin, end)

def Reduce_By_Key(ctx, key_in, value_in, key_out, value_out, binary_pred = None, binary_op = None, begin_key_in = 0, end_key_in = -1, begin_value_in = 0, begin_key_out = 0, begin_value_out = 0):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	cptr_binary_op = None
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	return native.n_reduce_by_key(ctx.m_cptr, key_in.m_cptr, value_in.m_cptr, key_out.m_cptr, value_out.m_cptr, cptr_binary_pred, cptr_binary_op, begin_key_in, end_key_in, begin_value_in, begin_key_out, begin_value_out)
