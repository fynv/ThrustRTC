#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "copy.h"
#include "remove.h"
#include "unique.h"
#include "partition.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	uint32_t Native::copy_if(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Copy_If(*vec_in, *vec_out, *pred, begin_in, end_in, begin_out);
	}

	uint32_t Native::copy_if_stencil(IntPtr p_vec_in, IntPtr p_vec_stencil, IntPtr p_vec_out, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_vec_stencil);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Copy_If_Stencil(*vec_in, *vec_stencil, *vec_out, *pred, begin_in, end_in, begin_stencil, begin_out);
	}

	uint32_t Native::remove(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		return TRTC_Remove(*vec, *value, begin, end);
	}

	uint32_t Native::remove_copy(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_value, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		return TRTC_Remove_Copy(*vec_in, *vec_out, *value, begin_in, end_in, begin_out);
	}

	uint32_t Native::remove_if(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Remove_If(*vec, *pred, begin, end);
	}

	uint32_t Native::remove_copy_if(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Remove_Copy_If(*vec_in, *vec_out, *pred, begin_in, end_in, begin_out);
	}

	uint32_t Native::remove_if_stencil(IntPtr p_vec, IntPtr p_stencil, IntPtr p_pred, size_t begin, size_t end, size_t begin_stencil)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* stencil = just_cast_it<DVVectorLike>(p_stencil);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Remove_If_Stencil(*vec, *stencil, *pred, begin, end, begin_stencil);
	}

	uint32_t Native::remove_copy_if_stencil(IntPtr p_vec_in, IntPtr p_stencil, IntPtr p_vec_out, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* stencil = just_cast_it<DVVectorLike>(p_stencil);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Remove_Copy_If_Stencil(*vec_in, *stencil, *vec_out, *pred, begin_in, end_in, begin_stencil, begin_out);
	}

	uint32_t Native::unique(IntPtr p_vec, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		return TRTC_Unique(*vec, begin, end);
	}

	uint32_t Native::unique(IntPtr p_vec, IntPtr p_binary_pred, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* binary_pred = just_cast_it<Functor>(p_binary_pred);
		return TRTC_Unique(*vec, *binary_pred, begin, end);
	}

	uint32_t Native::unique_copy(IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Unique_Copy(*vec_in, *vec_out, begin_in, end_in, begin_out);
	}

	uint32_t Native::unique_copy(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_binary_pred, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* binary_pred = just_cast_it<Functor>(p_binary_pred);
		return TRTC_Unique_Copy(*vec_in, *vec_out, *binary_pred, begin_in, end_in, begin_out);
	}

	uint32_t Native::unique_by_key(IntPtr p_keys, IntPtr p_values, size_t begin_key, size_t end_key, size_t begin_value)
	{
		DVVectorLike* keys = just_cast_it<DVVectorLike>(p_keys);
		DVVectorLike* values = just_cast_it<DVVectorLike>(p_values);
		return TRTC_Unique_By_Key(*keys, *values, begin_key, end_key, begin_value);
	}

	uint32_t Native::unique_by_key(IntPtr p_keys, IntPtr p_values, IntPtr p_binary_pred, size_t begin_key, size_t end_key, size_t begin_value)
	{
		DVVectorLike* keys = just_cast_it<DVVectorLike>(p_keys);
		DVVectorLike* values = just_cast_it<DVVectorLike>(p_values);
		Functor* binary_pred = just_cast_it<Functor>(p_binary_pred);
		return TRTC_Unique_By_Key(*keys, *values, *binary_pred, begin_key, end_key, begin_value);
	}

	uint32_t Native::unique_by_key_copy(IntPtr p_keys_in, IntPtr p_values_in, IntPtr p_keys_out, IntPtr p_values_out, size_t begin_key_in, size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out)
	{
		DVVectorLike* keys_in = just_cast_it<DVVectorLike>(p_keys_in);
		DVVectorLike* values_in = just_cast_it<DVVectorLike>(p_values_in);
		DVVectorLike* keys_out = just_cast_it<DVVectorLike>(p_keys_out);
		DVVectorLike* values_out = just_cast_it<DVVectorLike>(p_values_out);
		return TRTC_Unique_By_Key_Copy(*keys_in, *values_in, *keys_out, *values_out, begin_key_in, end_key_in, begin_value_in, begin_key_out, begin_value_out);
	}

	uint32_t Native::unique_by_key_copy(IntPtr p_keys_in, IntPtr p_values_in, IntPtr p_keys_out, IntPtr p_values_out, IntPtr p_binary_pred, size_t begin_key_in, size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out)
	{
		DVVectorLike* keys_in = just_cast_it<DVVectorLike>(p_keys_in);
		DVVectorLike* values_in = just_cast_it<DVVectorLike>(p_values_in);
		DVVectorLike* keys_out = just_cast_it<DVVectorLike>(p_keys_out);
		DVVectorLike* values_out = just_cast_it<DVVectorLike>(p_values_out);
		Functor* binary_pred = just_cast_it<Functor>(p_binary_pred);
		return TRTC_Unique_By_Key_Copy(*keys_in, *values_in, *keys_out, *values_out, *binary_pred, begin_key_in, end_key_in, begin_value_in, begin_key_out, begin_value_out);
	}

	uint32_t Native::partition(IntPtr p_vec, IntPtr p_pred, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Partition(*vec, *pred, begin, end);
	}

	uint32_t Native::partition_stencil(IntPtr p_vec, IntPtr p_stencil, IntPtr p_pred, size_t begin, size_t end, size_t begin_stencil)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DVVectorLike* stencil = just_cast_it<DVVectorLike>(p_stencil);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Partition_Stencil(*vec, *stencil, *pred, begin, end, begin_stencil);
	}

	uint32_t Native::partition_copy(IntPtr p_vec_in, IntPtr p_vec_true, IntPtr p_vec_false, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_true, size_t begin_false)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_true = just_cast_it<DVVectorLike>(p_vec_true);
		DVVectorLike* vec_false = just_cast_it<DVVectorLike>(p_vec_false);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Partition_Copy(*vec_in, *vec_true, *vec_false, *pred, begin_in, end_in, begin_true, begin_false);
	}

	uint32_t Native::partition_copy_stencil(IntPtr p_vec_in, IntPtr p_stencil, IntPtr p_vec_true, IntPtr p_vec_false, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_true, size_t begin_false)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_stencil);
		DVVectorLike* vec_true = just_cast_it<DVVectorLike>(p_vec_true);
		DVVectorLike* vec_false = just_cast_it<DVVectorLike>(p_vec_false);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Partition_Copy_Stencil(*vec_in, *vec_stencil, *vec_true, *vec_false, *pred, begin_in, end_in, begin_stencil, begin_true, begin_false);
	}

}
