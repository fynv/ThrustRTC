from . import PyThrustRTC as native

def Replace(ctx, vec, old_value, new_value, begin =0, end =-1):
	native.n_replace(ctx.m_cptr, vec.m_cptr, old_value.m_cptr, new_value.m_cptr, begin, end)

def Replace_If(ctx, vec, pred, new_value, begin =0, end =-1):
	native.n_replace_if(ctx.m_cptr, vec.m_cptr, pred, new_value.m_cptr, begin, end)

def Replace_Copy(ctx, vec_in, vec_out, old_value, new_value, begin_in =0, end_in =-1, begin_out=0):
	native.n_replace_copy(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, old_value.m_cptr, new_value.m_cptr, begin_in, end_in, begin_out)

def Replace_Copy_If(ctx, vec_in, vec_out, pred, new_value, begin_in =0, end_in =-1, begin_out=0):
	native.n_replace_copy_if(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, pred, new_value.m_cptr, begin_in, end_in, begin_out)
