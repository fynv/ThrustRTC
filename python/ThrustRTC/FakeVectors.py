from .Native import native, check_cptr
from .DVVector import DVVectorLike
from .utils import *
import ctypes

class DVConstant(DVVectorLike):
	def __init__(self, dvobj, size = -1):
		self.m_dvobj = dvobj
		self.m_cptr = check_cptr(native.n_dvconstant_create(dvobj.m_cptr, ctypes.c_ulonglong(size).value))

class DVCounter(DVVectorLike):
	def __init__(self, dvobj_init, size = -1):
		self.m_dvobj_init = dvobj_init
		self.m_cptr = check_cptr(native.n_dvcounter_create(dvobj_init.m_cptr, ctypes.c_ulonglong(size).value))

class DVDiscard(DVVectorLike):
	def __init__(self, elem_cls, size = -1):
		self.m_cptr = check_cptr(native.n_dvdiscard_create(elem_cls.encode('utf-8'), ctypes.c_ulonglong(size).value))

class DVPermutation(DVVectorLike):
	def __init__(self, vec_value, vec_index):
		self.m_vec_value = vec_value
		self.m_vec_index = vec_index
		self.m_cptr = check_cptr(native.n_dvpermutation_create(vec_value.m_cptr, vec_index.m_cptr))

class DVReverse(DVVectorLike):
	def __init__(self, vec_value):
		self.m_vec_value = vec_value
		self.m_cptr = check_cptr(native.n_dvreverse_create(vec_value.m_cptr))

class DVTransform(DVVectorLike):
	def __init__(self, vec_in, elem_cls, op):
		self.m_vec_in = vec_in
		self.m_op = op
		self.m_cptr = check_cptr(native.n_dvtransform_create(vec_in.m_cptr, elem_cls.encode('utf-8'), op.m_cptr))

class DVZipped(DVVectorLike):
	def __init__(self, vecs, elem_names):
		self.m_vecs = vecs
		o_vecs = ObjArray(vecs)
		o_elem_names = StrArray(elem_names)
		self.m_cptr = check_cptr(native.n_dvzipped_create(o_vecs.m_cptr, o_elem_names.m_cptr))

class DVCustomVector(DVVectorLike):
	def __init__(self, arg_map, name_idx, code_body, elem_cls, size = -1, read_only = True):
		self.m_arg_map = arg_map
		param_names = [param_name for param_name, elem in arg_map.items()]
		o_param_names = StrArray(param_names)
		elems = [elem for param_name, elem in arg_map.items()]
		o_elems = ObjArray(elems)

		self.m_cptr = check_cptr(native.n_dvcustomvector_create(
			o_elems.m_cptr, o_param_names.m_cptr,
			name_idx.encode('utf-8'), code_body.encode('utf-8'), 
			elem_cls.encode('utf-8'), ctypes.c_ulonglong(size).value, read_only))



