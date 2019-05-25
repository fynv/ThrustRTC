from . import PyThrustRTC as native

def Copy_If(ctx, vec_in, vec_out, pred, begin_in = 0, end_in = -1, begin_out = 0):
	return native.n_copy_if(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr, begin_in, end_in, begin_out)

def Copy_If_Stencil(ctx, vec_in, vec_stencil, vec_out, pred, begin_in = 0, end_in = -1, begin_stencil=0, begin_out = 0):
	return native.n_copy_if_stencil(ctx.m_cptr, vec_in.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, pred.m_cptr, begin_in, end_in, begin_stencil, begin_out)

def Remove(ctx, vec, value, begin = 0, end = -1):
	return native.n_remove(ctx.m_cptr, vec.m_cptr, value.m_cptr, begin, end)

def Remove_Copy(ctx, vec_in, vec_out, value, begin_in = 0, end_in = -1, begin_out = 0):
	return native.n_remove_copy(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, value.m_cptr, begin_in, end_in, begin_out)

def Remove_If(ctx, vec, pred, begin = 0, end = -1):
	return native.n_remove_if(ctx.m_cptr, vec.m_cptr, pred.m_cptr, begin, end)

def Remove_Copy_If(ctx, vec_in, vec_out, pred, begin_in = 0, end_in = -1, begin_out = 0):
	return native.n_remove_copy_if(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr, begin_in, end_in, begin_out)

def Remove_If_Stencil(ctx, vec, stencil, pred, begin = 0, end = -1, begin_stencil = 0):
	return native.n_remove_if_stencil(ctx.m_cptr, vec.m_cptr, stencil.m_cptr, pred.m_cptr, begin, end, begin_stencil)

def Remove_Copy_If_Stencil(ctx, vec_in, stencil, vec_out, pred, begin_in = 0, end_in = -1, begin_stencil = 0, begin_out = 0):
	return native.n_remove_copy_if_stencil(ctx.m_cptr, vec_in.m_cptr, stencil.m_cptr, vec_out.m_cptr, pred.m_cptr, begin_in, end_in, begin_stencil, begin_out)

def Unique(ctx, vec, binary_pred = None, begin = 0, end = -1):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	return native.n_unique(ctx.m_cptr, vec.m_cptr, cptr_binary_pred, begin, end)

def Unique_Copy(ctx, vec_in, vec_out, binary_pred = None, begin_in = 0, end_in = -1, begin_out = 0):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	return native.n_unique_copy(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, cptr_binary_pred, begin_in, end_in, begin_out)

def Unique_By_Key(ctx, keys, values, binary_pred = None, begin_key = 0, end_key = -1, begin_value = 0):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	return native.n_unique_by_key(ctx.m_cptr, keys.m_cptr, values.m_cptr, cptr_binary_pred, begin_key, end_key, begin_value)

def Unique_By_Key_Copy(ctx, keys_in, values_in, keys_out, values_out, binary_pred = None, begin_key_in = 0, end_key_in = -1, begin_value_in = 0, begin_key_out = 0, begin_value_out = 0):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr	
	return native.n_unique_by_key_copy(ctx.m_cptr, keys_in.m_cptr, values_in.m_cptr, keys_out.m_cptr, values_out.m_cptr, cptr_binary_pred, begin_key_in, end_key_in, begin_value_in, begin_key_out, begin_value_out)

def Partition(ctx, vec, pred, begin = 0, end = -1):
	return native.n_partition(ctx.m_cptr, vec.m_cptr, pred.m_cptr, begin, end)

def Partition_Stencil(ctx, vec, stencil, pred, begin = 0, end = -1, begin_stencil = 0):
	return native.n_partition_stencil(ctx.m_cptr, vec.m_cptr, stencil.m_cptr, pred.m_cptr, begin, end, begin_stencil)

def Partition_Copy(ctx, vec_in, vec_true, vec_false, pred, begin_in = 0, end_in = -1, begin_true = 0, begin_false = 0):
	return native.n_partition_copy(ctx.m_cptr, vec_in.m_cptr, vec_true.m_cptr, vec_false.m_cptr, pred.m_cptr, begin_in, end_in, begin_true, begin_false)

def Partition_Copy_Stencil(ctx, vec_in, stencil, vec_true, vec_false, pred, begin_in = 0, end_in = -1, begin_stencil = 0, begin_true = 0, begin_false = 0):
	return native.n_partition_copy_stencil(ctx.m_cptr, vec_in.m_cptr, stencil.m_cptr, vec_true.m_cptr, vec_false.m_cptr, pred.m_cptr, begin_in, end_in, begin_stencil, begin_true, begin_false)


