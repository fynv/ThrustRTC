from . import PyThrustRTC as native

def Fill(vec, value):
	native.n_fill(vec.m_cptr, value.m_cptr)

def For_Each(vec, f):
	native.n_for_each(vec.m_cptr, f.m_cptr)

def Replace(vec, old_value, new_value):
	native.n_replace(vec.m_cptr, old_value.m_cptr, new_value.m_cptr)

def Replace_If(vec, pred, new_value):
	native.n_replace_if(vec.m_cptr, pred.m_cptr, new_value.m_cptr)

def Replace_Copy(vec_in, vec_out, old_value, new_value):
	native.n_replace_copy(vec_in.m_cptr, vec_out.m_cptr, old_value.m_cptr, new_value.m_cptr)

def Replace_Copy_If(vec_in, vec_out, pred, new_value):
	native.n_replace_copy_if(vec_in.m_cptr, vec_out.m_cptr, pred.m_cptr, new_value.m_cptr)

def Adjacent_Difference(vec_in, vec_out, binary_op=None):
	cptr_binary_op = None
	if binary_op!=None:
		cptr_binary_op = binary_op.m_cptr
	native.n_adjacent_difference(vec_in.m_cptr, vec_out.m_cptr, cptr_binary_op,)

def Sequence(vec, value_init=None, value_step=None):
	cptr_init = None
	if value_init!=None:
		cptr_init = value_init.m_cptr
	cptr_step = None
	if value_step!=None:
		cptr_step = value_step.m_cptr
	native.n_sequence(vec.m_cptr, cptr_init, cptr_step)

def Tabulate(vec, op):
	native.n_tabulate(vec.m_cptr, op.m_cptr)

def Transform(vec_in, vec_out, op):
	native.n_transform(vec_in.m_cptr, vec_out.m_cptr, op.m_cptr)

def Transform_Binary(vec_in1, vec_in2, vec_out, op):
	native.n_transform_binary(vec_in1.m_cptr, vec_in2.m_cptr, vec_out.m_cptr, op.m_cptr)

def Transform_If(vec_in, vec_out, op, pred):
	native.n_transform_if(vec_in.m_cptr, vec_out.m_cptr, op.m_cptr, pred.m_cptr)

def Transform_If_Stencil(vec_in, vec_stencil, vec_out, op, pred):
	native.n_transform_if_stencil(vec_in.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, op.m_cptr, pred.m_cptr)

def Transform_Binary_If_Stencil(vec_in1, vec_in2, vec_stencil, vec_out, op, pred):
	native.n_transform_binary_if_stencil(vec_in1.m_cptr, vec_in2.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, op.m_cptr, pred.m_cptr)
