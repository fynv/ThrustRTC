from . import PyThrustRTC as native

def Inclusive_Scan(ctx, vec_in, vec_out, binary_op = None, begin_in = 0, end_in = -1, begin_out = 0):
	cptr_binary_op = None
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	native.n_inclusive_scan(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, cptr_binary_op, begin_in, end_in, begin_out)

def Exclusive_Scan(ctx, vec_in, vec_out, value_init = None, binary_op = None, begin_in = 0, end_in = -1, begin_out = 0):
	cptr_init = None
	if value_init!=None:
		cptr_init = value_init.m_cptr
	cptr_binary_op = None
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	native.n_exclusive_scan(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, cptr_init, cptr_binary_op, begin_in, end_in, begin_out)

def Transform_Inclusive_Scan(ctx, vec_in, vec_out, unary_op, binary_op, begin_in = 0, end_in = -1, begin_out = 0):
	native.n_transform_inclusive_scan(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, unary_op.m_cptr, binary_op.m_cptr, begin_in, end_in, begin_out)

def Transform_Exclusive_Scan(ctx, vec_in, vec_out, unary_op, value_init, binary_op, begin_in = 0, end_in = -1, begin_out = 0):
	native.n_transform_exclusive_scan(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, unary_op.m_cptr, value_init.m_cptr, binary_op.m_cptr, begin_in, end_in, begin_out)

def Inclusive_Scan_By_Key(ctx, vec_key, vec_value, vec_out, binary_pred = None, binary_op = None, begin_key = 0, end_key = -1, begin_value = 0, begin_out = 0):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	cptr_binary_op = None
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	native.n_inclusive_scan_by_key(ctx.m_cptr, vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, cptr_binary_pred, cptr_binary_op, begin_key, end_key, begin_value, begin_out)

def Exclusive_Scan_By_Key(ctx, vec_key, vec_value, vec_out, value_init = None, binary_pred = None, binary_op = None, begin_key = 0, end_key = -1, begin_value = 0, begin_out = 0):
	cptr_init = None
	if value_init!=None:
		cptr_init = value_init.m_cptr
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	cptr_binary_op = None
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	native.n_exclusive_scan_by_key(ctx.m_cptr, vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, cptr_init, cptr_binary_pred, cptr_binary_op, begin_key, end_key, begin_value, begin_out)
