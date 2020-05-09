#include "api.h"
#include "DVVector.h"
#include "fake_vectors/DVRange.h"

typedef std::vector<const DeviceViewable*> PtrArray;

const char* n_dvvectorlike_name_elem_cls(void* cptr)
{
	DVVectorLike* dvvec = (DVVectorLike*)cptr;
	return dvvec->name_elem_cls().c_str();
}

unsigned long long n_dvvectorlike_size(void* cptr)
{
	DVVectorLike* dvvec = (DVVectorLike*)cptr;
	return dvvec->size();
}

void* n_dvvector_create(const char* elem_cls, unsigned long long size, void* hdata)
{
	return new DVVector(elem_cls, size, hdata);
}

void n_dvvector_to_host(void* cptr, void* hdata, unsigned long long begin, unsigned long long end)
{
	DVVector* dvvec = (DVVector*)cptr;
	dvvec->to_host(hdata, begin, end);
}

void* n_dvvector_from_dvs(void* ptr_dvs)
{
	PtrArray* dvs = (PtrArray*)ptr_dvs;
	size_t num_items = dvs->size();
	if (num_items < 1) return nullptr;
	std::string elem_cls = (*dvs)[0]->name_view_cls();
	for (size_t i = 1; i < num_items; i++)
	{
		if ((*dvs)[i]->name_view_cls() != elem_cls)
			return nullptr;
	}
	size_t elem_size = TRTC_Size_Of(elem_cls.c_str());
	std::vector<char> buf(elem_size*num_items);
	for (size_t i = 0; i < num_items; i++)
	{
		memcpy(buf.data() + elem_size * i, (*dvs)[i]->view().data(), elem_size);
	}
	return new DVVector(elem_cls.data(), num_items, buf.data());
}

void* n_dvvectoradaptor_create(const char* elem_cls, unsigned long long size, void* data)
{
	return new DVVectorAdaptor(elem_cls, size, data);
}

void* n_dvrange_create(void* ptr_in, unsigned long long begin, unsigned long long end)
{
	DVVectorLike* vec_value = (DVVectorLike*)ptr_in;
	DVVector* p_vec = dynamic_cast<DVVector*>(vec_value);
	if (p_vec)
	{
		return new DVVectorAdaptor(*p_vec, begin, end);
	}

	DVVectorAdaptor* p_vec_adpt = dynamic_cast<DVVectorAdaptor*>(vec_value);
	if (p_vec_adpt)
	{
		return new DVVectorAdaptor(*p_vec_adpt, begin, end);
	}

	return new DVRange(*vec_value, begin, end);
}
