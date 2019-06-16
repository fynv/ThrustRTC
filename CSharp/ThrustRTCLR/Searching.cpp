#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "find.h"
#include "mismatch.h"
#include "binary_search.h"
#include "partition.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	Object^ Native::find(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		size_t result;
		if (!TRTC_Find(*vec, *value, result, begin, end))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::find_if(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		size_t result;
		if(!TRTC_Find_If(*vec, *pred, result, begin, end))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::find_if_not(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		size_t result;
		if (!TRTC_Find_If_Not(*vec, *pred, result, begin, end))
			return nullptr;
		return (int64_t)result;
	}

	Tuple<int64_t, int64_t>^ Native::mismatch(IntPtr p_vec1, IntPtr p_vec2, size_t begin1, size_t end1, size_t begin2)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		size_t result1, result2;
		if (!TRTC_Mismatch(*vec1, *vec2, result1, result2, begin1, end1, begin2))
			return nullptr;
		return gcnew Tuple<int64_t, int64_t>{ (int64_t)result1, (int64_t)result2};
	}

	Tuple<int64_t, int64_t>^ Native::mismatch(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_pred, size_t begin1, size_t end1, size_t begin2)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		Functor* pred = just_cast_it<Functor>(p_pred);
		size_t result1, result2;
		if (!TRTC_Mismatch(*vec1, *vec2, *pred, result1, result2, begin1, end1, begin2))
			return nullptr;
		return gcnew Tuple<int64_t, int64_t>{ (int64_t)result1, (int64_t)result2};
	}

	Object^ Native::lower_bound(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		size_t result;
		if (!TRTC_Lower_Bound(*vec, *value, result, begin, end))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::lower_bound(IntPtr p_vec, IntPtr p_value, IntPtr p_comp, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		Functor* comp = just_cast_it<Functor>(p_comp);
		size_t result;
		if (!TRTC_Lower_Bound(*vec, *value, *comp, result, begin, end))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::upper_bound(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		size_t result;
		if (!TRTC_Upper_Bound(*vec, *value, result, begin, end))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::upper_bound(IntPtr p_vec, IntPtr p_value, IntPtr p_comp, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		Functor* comp = just_cast_it<Functor>(p_comp);
		size_t result;
		if (!TRTC_Upper_Bound(*vec, *value, *comp, result, begin, end))
			return nullptr;
		return (int64_t)result;
	}

	Object^ Native::binary_search(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		bool result;
		if (!TRTC_Binary_Search(*vec, *value, result, begin, end))
			return nullptr;
		return result;
	}

	Object^ Native::binary_search(IntPtr p_vec, IntPtr p_value, IntPtr p_comp, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		Functor* comp = just_cast_it<Functor>(p_comp);
		bool result;
		if (!TRTC_Binary_Search(*vec, *value, *comp, result, begin, end))
			return nullptr;
		return result;
	}

	bool Native::lower_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		return TRTC_Lower_Bound_V(*vec, *value, *result, begin, end, begin_values, end_values, begin_result);
	}

	bool Native::lower_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, IntPtr p_comp, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Lower_Bound_V(*vec, *value, *result, *comp, begin, end, begin_values, end_values, begin_result);
	}

	bool Native::upper_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		return TRTC_Upper_Bound_V(*vec, *value, *result, begin, end, begin_values, end_values, begin_result);
	}

	bool Native::upper_bound_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, IntPtr p_comp, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Upper_Bound_V(*vec, *value, *result, *comp, begin, end, begin_values, end_values, begin_result);
	}

	bool Native::binary_search_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		return TRTC_Binary_Search_V(*vec, *value, *result, begin, end, begin_values, end_values, begin_result);
	}

	bool Native::binary_search_v(IntPtr p_vec, IntPtr p_values, IntPtr p_result, IntPtr p_comp, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* value = just_cast_it<DVVectorLike>(p_values);
		DVVectorLike* result = just_cast_it<DVVectorLike>(p_result);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Binary_Search_V(*vec, *value, *result, *comp, begin, end, begin_values, end_values, begin_result);
	}

	Object^ Native::partition_point(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		size_t result;
		if (!TRTC_Partition_Point(*vec, *pred, result, begin, end))
			return nullptr;
		return (int64_t)result;
	}
}
