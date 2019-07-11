import PyThrustRTC as native
from .DVVector import DVVectorLike

class DVConstant(DVVectorLike):
	def __init__(self, dvobj, size = -1):
		self.m_dvobj = dvobj
		self.m_cptr = native.n_dvconstant_create(dvobj.m_cptr, size)

class DVCounter(DVVectorLike):
	def __init__(self, dvobj_init, size = -1):
		self.m_dvobj_init = dvobj_init
		self.m_cptr = native.n_dvcounter_create(dvobj_init.m_cptr, size)

class DVDiscard(DVVectorLike):
	def __init__(self, elem_cls, size = -1):
		self.m_cptr = native.n_dvdiscard_create(elem_cls, size)

class DVPermutation(DVVectorLike):
	def __init__(self, vec_value, vec_index):
		self.m_vec_value = vec_value
		self.m_vec_index = vec_index
		self.m_cptr = native.n_dvpermutation_create(vec_value.m_cptr, vec_index.m_cptr)

class DVReverse(DVVectorLike):
	def __init__(self, vec_value):
		self.m_vec_value = vec_value
		self.m_cptr = native.n_dvreverse_create(vec_value.m_cptr)

class DVTransform(DVVectorLike):
	def __init__(self, vec_in, elem_cls, op):
		self.m_vec_in = vec_in
		self.m_op = op
		self.m_cptr = native.n_dvtransform_create(vec_in.m_cptr, elem_cls, op.m_cptr)

class DVZipped(DVVectorLike):
	def __init__(self, vecs, elem_names):
		self.m_vecs = vecs
		self.m_cptr = native.n_dvzipped_create([item.m_cptr for item in vecs], elem_names)

class DVCustomVector(DVVectorLike):
	def __init__(self, arg_map, name_idx, code_body, elem_cls, size = -1, read_only = True):
		self.m_arg_map = arg_map
		self.m_cptr = native.n_dvcustomvector_create(			
			[ (param_name, arg.m_cptr) for param_name, arg in arg_map.items()],
			name_idx, code_body, elem_cls, size, read_only)

