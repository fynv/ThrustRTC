#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "merge.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	bool Native::merge(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_vec_out, size_t begin1, size_t end1, size_t begin2, size_t end2, size_t begin_out)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Merge(*vec1, *vec2, *vec_out, begin1, end1, begin2, end2, begin_out);
	}

	bool Native::merge(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_vec_out, IntPtr p_comp, size_t begin1, size_t end1, size_t begin2, size_t end2, size_t begin_out)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Merge(*vec1, *vec2, *vec_out, *comp, begin1, end1, begin2, end2, begin_out);
	}

	bool Native::merge_by_key(IntPtr p_key1, IntPtr p_keys2, IntPtr p_value1, IntPtr p_value2, IntPtr p_keys_out, IntPtr p_value_out, size_t begin_keys1, size_t end_keys1, size_t begin_keys2, size_t end_keys2, size_t begin_value1, size_t begin_value2, size_t begin_keys_out, size_t begin_value_out)
	{
		DVVectorLike* keys1 = just_cast_it<DVVectorLike>(p_key1);
		DVVectorLike* keys2 = just_cast_it<DVVectorLike>(p_keys2);
		DVVectorLike* value1 = just_cast_it<DVVectorLike>(p_value1);
		DVVectorLike* value2 = just_cast_it<DVVectorLike>(p_value2);
		DVVectorLike* keys_out = just_cast_it<DVVectorLike>(p_keys_out);
		DVVectorLike* value_out = just_cast_it<DVVectorLike>(p_value_out);
		return TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out, begin_keys1, end_keys1, begin_keys2, end_keys2, begin_value1, begin_value2, begin_keys_out, begin_value_out);
	}

	bool Native::merge_by_key(IntPtr p_key1, IntPtr p_keys2, IntPtr p_value1, IntPtr p_value2, IntPtr p_keys_out, IntPtr p_value_out, IntPtr p_comp, size_t begin_keys1, size_t end_keys1, size_t begin_keys2, size_t end_keys2, size_t begin_value1, size_t begin_value2, size_t begin_keys_out, size_t begin_value_out)
	{
		DVVectorLike* keys1 = just_cast_it<DVVectorLike>(p_key1);
		DVVectorLike* keys2 = just_cast_it<DVVectorLike>(p_keys2);
		DVVectorLike* value1 = just_cast_it<DVVectorLike>(p_value1);
		DVVectorLike* value2 = just_cast_it<DVVectorLike>(p_value2);
		DVVectorLike* keys_out = just_cast_it<DVVectorLike>(p_keys_out);
		DVVectorLike* value_out = just_cast_it<DVVectorLike>(p_value_out);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out, *comp, begin_keys1, end_keys1, begin_keys2, end_keys2, begin_value1, begin_value2, begin_keys_out, begin_value_out);

	}
}
