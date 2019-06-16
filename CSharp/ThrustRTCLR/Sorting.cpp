#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "sort.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	bool Native::sort(IntPtr p_vec, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		return TRTC_Sort(*vec, begin, end);
	}

	bool Native::sort(IntPtr p_vec, IntPtr p_comp, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Sort(*vec, *comp, begin, end);
	}

	bool Native::sort_by_key(IntPtr p_keys, IntPtr p_values, size_t begin_keys, size_t end_keys, size_t begin_values)
	{
		DVVectorLike* keys = just_cast_it<DVVectorLike>(p_keys);
		DVVectorLike* values = just_cast_it<DVVectorLike>(p_values);
		return TRTC_Sort_By_Key(*keys, *values, begin_keys, end_keys, begin_values);
	}

	bool Native::sort_by_key(IntPtr p_keys, IntPtr p_values, IntPtr p_comp, size_t begin_keys, size_t end_keys, size_t begin_values)
	{
		DVVectorLike* keys = just_cast_it<DVVectorLike>(p_keys);
		DVVectorLike* values = just_cast_it<DVVectorLike>(p_values);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Sort_By_Key(*keys, *values, *comp, begin_keys, end_keys, begin_values);
	}
}