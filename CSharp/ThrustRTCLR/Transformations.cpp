#include "stdafx.h"
#include "ThrustRTCLR.h"

#include "fill.h"
#include "replace.h"
#include "for_each.h"
#include "adjacent_difference.h"
#include "sequence.h"
#include "tabulate.h"
#include "transform.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	bool Native::fiil(IntPtr p_vec, IntPtr p_value)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		return TRTC_Fill(*vec, *value);
	}

	bool Native::replace(IntPtr p_vec, IntPtr p_old_value, IntPtr p_new_value)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* old_value = just_cast_it<DeviceViewable>(p_old_value);
		DeviceViewable* new_value = just_cast_it<DeviceViewable>(p_new_value);
		return TRTC_Replace(*vec, *old_value, *new_value);
	}

	bool Native::replace_if(IntPtr p_vec, IntPtr p_pred, IntPtr p_new_value)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		DeviceViewable* new_value = just_cast_it<DeviceViewable>(p_new_value);
		return TRTC_Replace_If(*vec, *pred, *new_value);
	}

	bool Native::replace_copy(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_old_value, IntPtr p_new_value)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		DeviceViewable* old_value = just_cast_it<DeviceViewable>(p_old_value);
		DeviceViewable* new_value = just_cast_it<DeviceViewable>(p_new_value);
		return TRTC_Replace_Copy(*vec_in, *vec_out, *old_value, *new_value);
	}

	bool Native::replace_copy_if(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_pred, IntPtr p_new_value)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* pred = just_cast_it<Functor>(p_pred);
		DeviceViewable* new_value = just_cast_it<DeviceViewable>(p_new_value);
		return TRTC_Replace_Copy_If(*vec_in, *vec_out, *pred, *new_value);
	}

	bool Native::for_each(IntPtr p_vec, IntPtr p_f)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* f = just_cast_it<Functor>(p_f);
		return TRTC_For_Each(*vec, *f);
	}

	bool Native::adjacent_difference(IntPtr p_vec_in, IntPtr p_vec_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Adjacent_Difference(*vec_in, *vec_out);
	}

	bool Native::adjacent_difference(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_binary_op)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* binary_op = just_cast_it<Functor>(p_binary_op);
		return TRTC_Adjacent_Difference(*vec_in, *vec_out, *binary_op);
	}

	bool Native::sequence(IntPtr p_vec)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		return TRTC_Sequence(*vec);
	}

	bool Native::sequence(IntPtr p_vec, IntPtr p_value_init)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value_init = just_cast_it<DeviceViewable>(p_value_init);
		return TRTC_Sequence(*vec, *value_init);
	}

	bool Native::sequence(IntPtr p_vec, IntPtr p_value_init, IntPtr p_value_step)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value_init = just_cast_it<DeviceViewable>(p_value_init);
		DeviceViewable* value_step = just_cast_it<DeviceViewable>(p_value_step);
		return TRTC_Sequence(*vec, *value_init, *value_step);
	}

	bool Native::tabulate(IntPtr p_vec, IntPtr p_op)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* op = just_cast_it<Functor>(p_op);
		return TRTC_Tabulate(*vec, *op);
	}

	bool Native::transform(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_op)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* op = just_cast_it<Functor>(p_op);
		return TRTC_Transform(*vec_in, *vec_out, *op);
	}

	bool Native::transform_binary(IntPtr p_vec_in1, IntPtr p_vec_in2, IntPtr p_vec_out, IntPtr p_op)
	{
		DVVectorLike* vec_in1 = just_cast_it<DVVectorLike>(p_vec_in1);
		DVVectorLike* vec_in2 = just_cast_it<DVVectorLike>(p_vec_in2);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* op = just_cast_it<Functor>(p_op);
		return TRTC_Transform_Binary(*vec_in1, *vec_in2, *vec_out, *op);
	}

	bool Native::transform_if(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_op, IntPtr p_pred)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* op = just_cast_it<Functor>(p_op);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Transform_If(*vec_in, *vec_out, *op, *pred);
	}

	bool Native::transform_if_stencil(IntPtr p_vec_in, IntPtr p_vec_stencil, IntPtr p_vec_out, IntPtr p_op, IntPtr p_pred)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_vec_stencil);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* op = just_cast_it<Functor>(p_op);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Transform_If_Stencil(*vec_in, *vec_stencil, *vec_out, *op, *pred);
	}

	bool Native::transform_binary_if_stencil(IntPtr p_vec_in1, IntPtr p_vec_in2, IntPtr p_vec_stencil, IntPtr p_vec_out, IntPtr p_op, IntPtr p_pred)
	{
		DVVectorLike* vec_in1 = just_cast_it<DVVectorLike>(p_vec_in1);
		DVVectorLike* vec_in2 = just_cast_it<DVVectorLike>(p_vec_in2);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_vec_stencil);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* op = just_cast_it<Functor>(p_op);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Transform_Binary_If_Stencil(*vec_in1, *vec_in2, *vec_stencil, *vec_out, *op, *pred);
	}

}

