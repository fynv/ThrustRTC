from . import PyThrustRTC as native
from .DVVector import DVVectorLike

class DVConstant(DVVectorLike):
	def __init__(self, ctx, dvobj, size = -1):
		self.m_cptr = native.n_dvconstant_create(ctx.m_cptr, dvobj.m_cptr, size)