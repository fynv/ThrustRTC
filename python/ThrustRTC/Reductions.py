from . import PyThrustRTC as native

def Count(ctx, vec, value, begin = 0, end = -1):
	return native.n_count(ctx.m_cptr, vec.m_cptr, value.m_cptr, begin, end)

def Count_If(ctx, vec, pred, begin = 0, end = -1):
	return native.n_count_if(ctx.m_cptr, vec.m_cptr, pred, begin, end)

def Reduce(ctx, vec, value_init=None, binary_op=None, begin = 0, end = -1):
	cptr_init = None
	if value_init!=None:
		cptr_init = value_init.m_cptr
	return native.n_reduce(ctx.m_cptr, vec.m_cptr, cptr_init, binary_op, begin, end)

def Equal(ctx, vec1, vec2, binary_pred=None, begin1 = 0, end1 = -1, begin2 = 0 ):
	return native.n_equal(ctx.m_cptr, vec1.m_cptr, vec2.m_cptr, binary_pred, begin1, end1, begin2)

def Min_Element(ctx, vec, comp = None, begin = 0, end = -1):
	return native.n_min_element(ctx.m_cptr, vec.m_cptr, comp, begin, end)

def Max_Element(ctx, vec, comp = None, begin = 0, end = -1):
	return native.n_max_element(ctx.m_cptr, vec.m_cptr, comp, begin, end)

def MinMax_Element(ctx, vec, comp = None, begin = 0, end = -1):
	return native.n_minmax_element(ctx.m_cptr, vec.m_cptr, comp, begin, end)

def Inner_Product(ctx, vec1, vec2, value_init, binary_op1 = None, binary_op2 = None, begin1 = 0, end1 = -1, begin2 = 0):
	return native.n_inner_product(ctx.m_cptr, vec1.m_cptr, vec2.m_cptr, value_init.m_cptr, binary_op1, binary_op2, begin1, end1, begin2)

def Transform_Reduce(ctx, vec, unary_op, value_init, binary_op, begin = 0, end = -1):
	return native.n_transform_reduce(ctx.m_cptr, vec.m_cptr, unary_op, value_init.m_cptr, binary_op, begin, end)