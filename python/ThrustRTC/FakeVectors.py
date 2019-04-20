from . import PyThrustRTC as native
from .DVVector import DVVectorLike

class DVConstant(DVVectorLike):
	def __init__(self, ctx, dvobj, size = -1):
		self.m_cptr = native.n_dvconstant_create(ctx.m_cptr, dvobj.m_cptr, size)

class DVCounter(DVVectorLike):
	def __init__(self, ctx, dvobj_init, size = -1):
		self.m_cptr = native.n_dvcounter_create(ctx.m_cptr, dvobj_init.m_cptr, size)

class DVDiscard(DVVectorLike):
	def __init__(self, ctx, elem_cls, size = -1):
		self.m_cptr = native.n_dvdiscard_create(ctx.m_cptr, elem_cls, size)
