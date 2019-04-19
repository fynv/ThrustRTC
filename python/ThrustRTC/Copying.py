from . import PyThrustRTC as native

def Gather(ctx, vec_map, vec_in, vec_out, begin_map = 0, end_map =-1, begin_in = 0, begin_out = 0):
	native.n_gather(ctx.m_cptr, vec_map.m_cptr, vec_in.m_cptr, vec_out.m_cptr, begin_map, end_map, begin_in, begin_out)

def Gather_If(ctx, vec_map, vec_stencil, vec_in, vec_out, pred=None, begin_map = 0, end_map =-1, begin_stencil = 0, begin_in = 0, begin_out = 0):
	native.n_gather_if(ctx.m_cptr, vec_map.m_cptr, vec_stencil.m_cptr, vec_in.m_cptr, vec_out.m_cptr, pred, begin_map, end_map, begin_stencil, begin_in, begin_out)

