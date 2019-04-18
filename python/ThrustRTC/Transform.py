from . import PyThrustRTC as native

def Transform(ctx, vec_in, vec_out, op, begin_in = 0, end_in = -1, begin_out = 0):
	native.n_transform(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, op, begin_in, end_in, begin_out)

def Transform_Binary(ctx, vec_in1, vec_in2, vec_out, op, begin_in1 = 0, end_in1 = -1, begin_in2 = 0, begin_out = 0):
	native.n_transform_binary(ctx.m_cptr, vec_in1.m_cptr, vec_in2.m_cptr, vec_out.m_cptr, op, begin_in1, end_in1, begin_in2, begin_out)

def Transform_If(ctx, vec_in, vec_out, op, pred, begin_in = 0, end_in = -1, begin_out = 0):
	native.n_transform_if(ctx.m_cptr, vec_in.m_cptr, vec_out.m_cptr, op, pred, begin_in, end_in, begin_out)

def Transform_If_Stencil(ctx, vec_in, vec_stencil, vec_out, op, pred, begin_in = 0, end_in = -1, begin_stencil=0, begin_out = 0):
	native.n_transform_if_stencil(ctx.m_cptr, vec_in.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, op, pred, begin_in, end_in, begin_stencil, begin_out)

def Transform_Binary_If_Stencil(ctx, vec_in1, vec_in2, vec_stencil, vec_out, op, pred, begin_in1 = 0, end_in1 = -1, begin_in2 = 0, begin_stencil=0, begin_out = 0):
	native.n_transform_binary_if_stencil(ctx.m_cptr, vec_in1.m_cptr, vec_in2.m_cptr, vec_stencil.m_cptr, vec_out.m_cptr, op, pred, begin_in1, end_in1, begin_in2, begin_stencil, begin_out)
