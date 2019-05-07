from . import PyThrustRTC as native

def Gather(ctx, vec_map, vec_in, vec_out, begin_map = 0, end_map = -1, begin_in = 0, begin_out = 0):
	native.n_gather(ctx.m_cptr, vec_map.m_cptr, vec_in.m_cptr, vec_out.m_cptr, begin_map, end_map, begin_in, begin_out)

def Gather_If(ctx, vec_map, vec_stencil, vec_in, vec_out, pred = None, begin_map = 0, end_map =-1, begin_stencil = 0, begin_in = 0, begin_out = 0):
	cptr_pred = None
	if pred!=None:
		cptr_pred = pred.m_cptr
	native.n_gather_if(ctx.m_cptr, vec_map.m_cptr, vec_stencil.m_cptr, vec_in.m_cptr, vec_out.m_cptr, cptr_pred, begin_map, end_map, begin_stencil, begin_in, begin_out)

def Scatter(ctx, vec_in, vec_map, vec_out, begin_in = 0, end_in = -1, begin_map = 0, begin_out = 0):
	native.n_scatter(ctx.m_cptr, vec_in.m_cptr, vec_map.m_cptr, vec_out.m_cptr, begin_in, end_in, begin_map, begin_out)

def Scatter_If(ctx, vec_in, vec_map, vec_stencil, vec_out, pred = None, begin_in = 0, end_in = -1, begin_map = 0, begin_stencil = 0, begin_out = 0):
	cptr_pred = None
	if pred!=None:
		cptr_pred = pred.m_cptr
	native.n_scatter_if(ctx.m_cptr, vec_in.m_cptr, vec_map.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, cptr_pred, begin_in, end_in, begin_map, begin_stencil, begin_out)

def Copy(ctx, vec_in, vec_out, begin_in = 0, end_in = -1, begin_out = 0):
	native.n_copy(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, begin_in, end_in, begin_out)

def Swap(ctx, vec1, vec2, begin1 = 0, end1 = -1, begin2 = 0):
	native.n_swap(ctx.m_cptr, vec1.m_cptr, vec2.m_cptr, begin1, end1, begin2)	
