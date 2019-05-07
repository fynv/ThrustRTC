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
