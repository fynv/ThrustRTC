#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "find.h"
#include "mismatch.h"
#include "binary_search.h"
#include "partition.h"
#include "sort.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	Object^ Native::find(IntPtr p_vec, IntPtr p_value)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		size_t result;
		if (!TRTC_Find(*vec, *value, result))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::find_if(IntPtr p_vec, IntPtr p_pred)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		size_t result;
		if(!TRTC_Find_If(*vec, *pred, result))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::find_if_not(IntPtr p_vec, IntPtr p_pred)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		size_t result;
		if (!TRTC_Find_If_Not(*vec, *pred, result))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::mismatch(IntPtr p_vec1, IntPtr p_vec2)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		size_t result;
		if (!TRTC_Mismatch(*vec1, *vec2, result))
			return nullptr;
		return (int64_t)result;
	}

	Object^  Native::mismatch(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_pred)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		Functor* pred = just_cast_it<Functor>(p_pred);
		size_t result;
		if (!TRTC_Mismatch(*vec1, *vec2, *pred, result))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::lower_bound(IntPtr p_vec, IntPtr p_value)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		size_t result;
		if (!TRTC_Lower_Bound(*vec, *value, result))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::lower_bound(IntPtr p_vec, IntPtr p_value, IntPtr p_comp)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		Functor* comp = just_cast_it<Functor>(p_comp);
		size_t result;
		if (!TRTC_Lower_Bound(*vec, *value, *comp, result))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::upper_bound(IntPtr p_vec, IntPtr p_value)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		size_t result;
		if (!TRTC_Upper_Bound(*vec, *value, result))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::upper_bound(IntPtr p_vec, IntPtr p_value, IntPtr p_comp)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		Functor* comp = just_cast_it<Functor>(p_comp);
		size_t result;
		if (!TRTC_Upper_Bound(*vec, *value, *comp, result))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::binary_search(IntPtr p_vec, IntPtr p_value)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		bool result;
		if (!TRTC_Binary_Search(*vec, *value, result))
			return nullptr;
		return result;
	}

	Object^ Native::binary_search(IntPtr p_vec, IntPtr p_value, IntPtr p_comp)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		Functor* comp = just_cast_it<Functor>(p_comp);
		bool result;
		if (!TRTC_Binary_Search(*vec, *value, *comp, result))
			return nullptr;
		return result;
	}

	bool Native::lower_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		return TRTC_Lower_Bound_V(*vec, *value, *result);
	}

	bool Native::lower_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, IntPtr p_comp)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Lower_Bound_V(*vec, *value, *result, *comp);
	}

	bool Native::upper_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		return TRTC_Upper_Bound_V(*vec, *value, *result);
	}

	bool Native::upper_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, IntPtr p_comp)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Upper_Bound_V(*vec, *value, *result, *comp);
	}

	bool Native::binary_search_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		return TRTC_Binary_Search_V(*vec, *value, *result);
	}

	bool Native::binary_search_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, IntPtr p_comp)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Binary_Search_V(*vec, *value, *result, *comp);
	}

	Object^ Native::partition_point(IntPtr p_vec, IntPtr p_pred)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		size_t result;
		if (!TRTC_Partition_Point(*vec, *pred, result))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::is_sorted_until(IntPtr p_vec)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		size_t result;
		if (!TRTC_Is_Sorted_Until(*vec, result))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::is_sorted_until(IntPtr p_vec, IntPtr p_comp)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* comp = just_cast_it<Functor>(p_comp);
		size_t result;
		if (!TRTC_Is_Sorted_Until(*vec, *comp, result))
			return nullptr;
		return (int64_t)result;
	}
}
