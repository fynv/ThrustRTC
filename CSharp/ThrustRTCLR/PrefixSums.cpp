#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "scan.h"
#include "transform_scan.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	bool Native::inclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Inclusive_Scan(*vec_in, *vec_out);
	}

	bool Native::inclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_binary_op)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* binary_op = just_cast_it<Functor>(p_binary_op);
		return TRTC_Inclusive_Scan(*vec_in, *vec_out, *binary_op);
	}

	bool Native::exclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Exclusive_Scan(*vec_in, *vec_out);
	}

	bool Native::exclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_init)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		DeviceViewable* init = just_cast_it<DeviceViewable>(p_init);
		return TRTC_Exclusive_Scan(*vec_in, *vec_out, *init);
	}

	bool Native::exclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_init, IntPtr p_binary_op)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		DeviceViewable* init = just_cast_it<DeviceViewable>(p_init);
		Functor* binary_op = just_cast_it<Functor>(p_binary_op);
		return TRTC_Exclusive_Scan(*vec_in, *vec_out, *init, *binary_op);
	}

	bool Native::inclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out)
	{
		DVVectorLike* vec_key = just_cast_it<DVVectorLike>(p_vec_key);
		DVVectorLike* vec_value = just_cast_it<DVVectorLike>(p_vec_value);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out);
	}

	bool Native::inclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, IntPtr p_binary_pred)
	{
		DVVectorLike* vec_key = just_cast_it<DVVectorLike>(p_vec_key);
		DVVectorLike* vec_value = just_cast_it<DVVectorLike>(p_vec_value);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* binary_pred = just_cast_it<Functor>(p_binary_pred);
		return TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *binary_pred);
	}

	bool Native::inclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, IntPtr p_binary_pred, IntPtr p_binary_op)
	{
		DVVectorLike* vec_key = just_cast_it<DVVectorLike>(p_vec_key);
		DVVectorLike* vec_value = just_cast_it<DVVectorLike>(p_vec_value);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* binary_pred = just_cast_it<Functor>(p_binary_pred);
		Functor* binary_op = just_cast_it<Functor>(p_binary_op);
		return TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *binary_pred, *binary_op);
	}

	bool Native::exclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out)
	{
		DVVectorLike* vec_key = just_cast_it<DVVectorLike>(p_vec_key);
		DVVectorLike* vec_value = just_cast_it<DVVectorLike>(p_vec_value);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out);
	}

	bool Native::exclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, IntPtr p_init)
	{
		DVVectorLike* vec_key = just_cast_it<DVVectorLike>(p_vec_key);
		DVVectorLike* vec_value = just_cast_it<DVVectorLike>(p_vec_value);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		DeviceViewable* init = just_cast_it<DVVectorLike>(p_init);
		return TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init);
	}

	bool Native::exclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, IntPtr p_init, IntPtr p_binary_pred)
	{
		DVVectorLike* vec_key = just_cast_it<DVVectorLike>(p_vec_key);
		DVVectorLike* vec_value = just_cast_it<DVVectorLike>(p_vec_value);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		DeviceViewable* init = just_cast_it<DVVectorLike>(p_init);
		Functor* binary_pred = just_cast_it<Functor>(p_binary_pred);
		return TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init, *binary_pred);
	}

	bool Native::exclusive_scan_by_key(IntPtr p_vec_key, IntPtr p_vec_value, IntPtr p_vec_out, IntPtr p_init, IntPtr p_binary_pred, IntPtr p_binary_op)
	{
		DVVectorLike* vec_key = just_cast_it<DVVectorLike>(p_vec_key);
		DVVectorLike* vec_value = just_cast_it<DVVectorLike>(p_vec_value);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		DeviceViewable* init = just_cast_it<DVVectorLike>(p_init);
		Functor* binary_pred = just_cast_it<Functor>(p_binary_pred);
		Functor* binary_op = just_cast_it<Functor>(p_binary_op);
		return TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init, *binary_pred, *binary_op);
	}

	bool Native::transform_inclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_unary_op, IntPtr p_binary_op)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* unary_op = just_cast_it<Functor>(p_unary_op);
		Functor* binary_op = just_cast_it<Functor>(p_binary_op);
		return TRTC_Transform_Inclusive_Scan(*vec_in, *vec_out, *unary_op, *binary_op);
	}

	bool Native::transform_exclusive_scan(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_unary_op, IntPtr p_init, IntPtr p_binary_op)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* unary_op = just_cast_it<Functor>(p_unary_op);
		DeviceViewable* init = just_cast_it<DeviceViewable>(p_init);
		Functor* binary_op = just_cast_it<Functor>(p_binary_op);
		return TRTC_Transform_Exclusive_Scan(*vec_in, *vec_out, *unary_op, *init, *binary_op);
	}

}
