from . import PyThrustRTC as native

def Copy_If(vec_in, vec_out, pred):
	return native.n_copy_if(vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr)

def Copy_If_Stencil(vec_in, vec_stencil, vec_out, pred):
	return native.n_copy_if_stencil(vec_in.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, pred.m_cptr)

def Remove(vec, value):
	return native.n_remove(vec.m_cptr, value.m_cptr)

def Remove_Copy(vec_in, vec_out, value):
	return native.n_remove_copy(vec_in.m_cptr, vec_out.m_cptr, value.m_cptr)

def Remove_If(vec, pred):
	return native.n_remove_if(vec.m_cptr, pred.m_cptr)

def Remove_Copy_If(vec_in, vec_out, pred):
	return native.n_remove_copy_if(vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr)

def Remove_If_Stencil(vec, stencil, pred):
	return native.n_remove_if_stencil(vec.m_cptr, stencil.m_cptr, pred.m_cptr)

def Remove_Copy_If_Stencil(vec_in, stencil, vec_out, pred):
	return native.n_remove_copy_if_stencil(vec_in.m_cptr, stencil.m_cptr, vec_out.m_cptr, pred.m_cptr)

def Unique(vec, binary_pred = None):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	return native.n_unique(vec.m_cptr, cptr_binary_pred)

def Unique_Copy(vec_in, vec_out, binary_pred = None):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	return native.n_unique_copy(vec_in.m_cptr, vec_out.m_cptr, cptr_binary_pred)

def Unique_By_Key(keys, values, binary_pred = None):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	return native.n_unique_by_key(keys.m_cptr, values.m_cptr, cptr_binary_pred)

def Unique_By_Key_Copy(keys_in, values_in, keys_out, values_out, binary_pred = None):
	cptr_binary_pred = None
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr	
	return native.n_unique_by_key_copy(keys_in.m_cptr, values_in.m_cptr, keys_out.m_cptr, values_out.m_cptr, cptr_binary_pred)

def Partition(vec, pred):
	return native.n_partition(vec.m_cptr, pred.m_cptr)

def Partition_Stencil(vec, stencil, pred):
	return native.n_partition_stencil(vec.m_cptr, stencil.m_cptr, pred.m_cptr)

def Partition_Copy(vec_in, vec_true, vec_false, pred):
	return native.n_partition_copy(vec_in.m_cptr, vec_true.m_cptr, vec_false.m_cptr, pred.m_cptr)

def Partition_Copy_Stencil(vec_in, stencil, vec_true, vec_false, pred):
	return native.n_partition_copy_stencil(vec_in.m_cptr, stencil.m_cptr, vec_true.m_cptr, vec_false.m_cptr, pred.m_cptr)


