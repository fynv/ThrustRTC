
#include "ThrustRTCLR.h"
#include "merge.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	bool Native::merge(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_vec_out)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Merge(*vec1, *vec2, *vec_out);
	}

	bool Native::merge(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_vec_out, IntPtr p_comp)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Merge(*vec1, *vec2, *vec_out, *comp);
	}

	bool Native::merge_by_key(IntPtr p_keys1, IntPtr p_keys2, IntPtr p_value1, IntPtr p_value2, IntPtr p_keys_out, IntPtr p_value_out)
	{
		DVVectorLike* keys1 = just_cast_it<DVVectorLike>(p_keys1);
		DVVectorLike* keys2 = just_cast_it<DVVectorLike>(p_keys2);
		DVVectorLike* value1 = just_cast_it<DVVectorLike>(p_value1);
		DVVectorLike* value2 = just_cast_it<DVVectorLike>(p_value2);
		DVVectorLike* keys_out = just_cast_it<DVVectorLike>(p_keys_out);
		DVVectorLike* value_out = just_cast_it<DVVectorLike>(p_value_out);
		return TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out);
	}

	bool Native::merge_by_key(IntPtr p_key1, IntPtr p_keys2, IntPtr p_value1, IntPtr p_value2, IntPtr p_keys_out, IntPtr p_value_out, IntPtr p_comp)
	{
		DVVectorLike* keys1 = just_cast_it<DVVectorLike>(p_key1);
		DVVectorLike* keys2 = just_cast_it<DVVectorLike>(p_keys2);
		DVVectorLike* value1 = just_cast_it<DVVectorLike>(p_value1);
		DVVectorLike* value2 = just_cast_it<DVVectorLike>(p_value2);
		DVVectorLike* keys_out = just_cast_it<DVVectorLike>(p_keys_out);
		DVVectorLike* value_out = just_cast_it<DVVectorLike>(p_value_out);
		Functor* comp = just_cast_it<Functor>(p_comp);
		return TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out, *comp);
	}
}
