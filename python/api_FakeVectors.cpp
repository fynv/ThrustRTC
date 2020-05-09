#include "api.h"
#include "TRTCContext.h"
#include "fake_vectors/DVConstant.h"
#include "fake_vectors/DVCounter.h"
#include "fake_vectors/DVDiscard.h"
#include "fake_vectors/DVPermutation.h"
#include "fake_vectors/DVReverse.h"
#include "fake_vectors/DVTransform.h"
#include "fake_vectors/DVZipped.h"
#include "fake_vectors/DVCustomVector.h"

typedef std::vector<const DeviceViewable*> PtrArray;
typedef std::vector<std::string> StrArray;

void* n_dvconstant_create(void* cptr, unsigned long long size)
{
	DeviceViewable* dvobj = (DeviceViewable*)cptr;
	return new DVConstant(*dvobj, size);	
}

void* n_dvcounter_create(void* cptr, unsigned long long size)
{
	DeviceViewable* dvobj_init = (DeviceViewable*)cptr;
	return new DVCounter(*dvobj_init, size);	
}

void* n_dvdiscard_create(const char* elem_cls, unsigned long long size)
{
	return new DVDiscard(elem_cls, size);
}

void* n_dvpermutation_create(void* ptr_value, void* ptr_index)
{
	DVVectorLike* vec_value = (DVVectorLike*)ptr_value;
	DVVectorLike* vec_index = (DVVectorLike*)ptr_index;
	return new DVPermutation(*vec_value, *vec_index);
}

void* n_dvreverse_create(void* ptr_value)
{
	DVVectorLike* vec_value = (DVVectorLike*)ptr_value;
	return new DVReverse(*vec_value);
}

void* n_dvtransform_create(void* ptr_in, const char* elem_cls, void* ptr_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	Functor* op = (Functor*)ptr_op;
	return new DVTransform(*vec_in, elem_cls, *op);
}

void* n_dvzipped_create(void* ptr_vecs, void* ptr_elem_names)
{
	std::vector<DVVectorLike*>* vecs = (std::vector<DVVectorLike*>*)ptr_vecs;
	size_t num_vecs = vecs->size();
	StrArray* str_elem_names = (StrArray*)ptr_elem_names;
	size_t num_elems = str_elem_names->size();
	if (num_elems != num_vecs)
	{
		printf("Number of vectors %d mismatch with number of element names %d.", (int)num_vecs, (int)num_elems);
		return nullptr;
	}
	std::vector<const char*> elem_names(num_elems);
	for (size_t i = 0; i < num_elems; i++)
		elem_names[i] = (*str_elem_names)[i].c_str();
	return new DVZipped(*vecs, elem_names);
}

void* n_dvcustomvector_create(void* ptr_dvs, void* ptr_names, const char* name_idx, const char* body, const char* elem_cls, unsigned long long size, unsigned read_only)
{
	PtrArray* dvs = (PtrArray*)ptr_dvs;
	StrArray* names = (StrArray*)ptr_names;
	size_t num_params = dvs->size();
	std::vector<CapturedDeviceViewable> arg_map(num_params);
	for (size_t i = 0; i < num_params; i++)
	{
		arg_map[i].obj_name = (*names)[i].c_str();
		arg_map[i].obj = (*dvs)[i];
	}	
	return new DVCustomVector(arg_map, name_idx, body, elem_cls, size, read_only!=0);
}








