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