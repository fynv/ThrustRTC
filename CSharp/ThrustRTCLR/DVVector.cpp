#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "DVVector.h"
#include "fake_vectors/DVRange.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	String^ Native::dvvectorlike_name_elem_cls(IntPtr p_dvvec)
	{
		DVVectorLike* dvvec = just_cast_it<DVVectorLike>(p_dvvec);
		return gcnew String(dvvec->name_elem_cls().c_str());
	}

	size_t Native::dvvectorlike_size(IntPtr p_dvvec)
	{
		DVVectorLike* dvvec = just_cast_it<DVVectorLike>(p_dvvec);
		return dvvec->size();
	}

	IntPtr Native::dvvector_create(IntPtr p_elem_cls, size_t size, IntPtr p_hdata)
	{
		const char* elem_cls = just_cast_it<const char>(p_elem_cls);
		DVVector* cptr = new DVVector(elem_cls, size, (void*)p_hdata);
		return IntPtr(cptr);
	}

	void Native::dvvector_to_host(IntPtr p_dvvec, IntPtr p_hdata, size_t begin, size_t end)
	{
		DVVector* dvvec = just_cast_it<DVVector>(p_dvvec);
		dvvec->to_host((void*)p_hdata, begin, end);
	}

	IntPtr Native::dvrange_create(IntPtr p_vec_value, size_t begin, size_t end)
	{
		DVVectorLike* vec_value = just_cast_it<DVVectorLike>(p_vec_value);
		
		DVVector* p_vec = dynamic_cast<DVVector*>(vec_value);
		if (p_vec)
			return (IntPtr)(new DVVectorAdaptor(*p_vec, begin, end));
		
		DVVectorAdaptor* p_vec_adpt = dynamic_cast<DVVectorAdaptor*>(vec_value);
		if (p_vec_adpt)
			return (IntPtr)(new DVVectorAdaptor(*p_vec_adpt, begin, end));

		return (IntPtr)(new DVRange(*vec_value, begin, end));
	}
}
