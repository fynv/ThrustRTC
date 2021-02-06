from .Native import ffi, native, check_i

def Inclusive_Scan(vec_in, vec_out, binary_op = None):
	cptr_binary_op = ffi.NULL
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	check_i(native.n_inclusive_scan(vec_in.m_cptr, vec_out.m_cptr, cptr_binary_op))

def Exclusive_Scan(vec_in, vec_out, value_init = None, binary_op = None):
	cptr_init = ffi.NULL
	if value_init!=None:
		cptr_init = value_init.m_cptr
	cptr_binary_op = ffi.NULL
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	check_i(native.n_exclusive_scan(vec_in.m_cptr, vec_out.m_cptr, cptr_init, cptr_binary_op))

def Transform_Inclusive_Scan(vec_in, vec_out, unary_op, binary_op):
	check_i(native.n_transform_inclusive_scan(vec_in.m_cptr, vec_out.m_cptr, unary_op.m_cptr, binary_op.m_cptr))

def Transform_Exclusive_Scan(vec_in, vec_out, unary_op, value_init, binary_op):
	check_i(native.n_transform_exclusive_scan(vec_in.m_cptr, vec_out.m_cptr, unary_op.m_cptr, value_init.m_cptr, binary_op.m_cptr))

def Inclusive_Scan_By_Key(vec_key, vec_value, vec_out, binary_pred = None, binary_op = None):
	cptr_binary_pred = ffi.NULL
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	cptr_binary_op = ffi.NULL
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	check_i(native.n_inclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, cptr_binary_pred, cptr_binary_op))

def Exclusive_Scan_By_Key(vec_key, vec_value, vec_out, value_init = None, binary_pred = None, binary_op = None):
	cptr_init = ffi.NULL
	if value_init!=None:
		cptr_init = value_init.m_cptr
	cptr_binary_pred = ffi.NULL
	if binary_pred!=None:
		cptr_binary_pred = binary_pred.m_cptr
	cptr_binary_op = ffi.NULL
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	check_i(native.n_exclusive_scan_by_key(vec_key.m_cptr, vec_value.m_cptr, vec_out.m_cptr, cptr_init, cptr_binary_pred, cptr_binary_op))
