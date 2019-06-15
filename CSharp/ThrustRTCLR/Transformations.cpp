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

	bool Native::fiil(IntPtr p_vec, IntPtr p_value, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		return TRTC_Fill(*vec, *value, begin, end);
	}

	bool Native::replace(IntPtr p_vec, IntPtr p_old_value, IntPtr p_new_value, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* old_value = just_cast_it<DeviceViewable>(p_old_value);
		DeviceViewable* new_value = just_cast_it<DeviceViewable>(p_new_value);
		return TRTC_Replace(*vec, *old_value, *new_value, begin, end);
	}

	bool Native::replace_if(IntPtr p_vec, IntPtr p_pred, IntPtr p_new_value, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		DeviceViewable* new_value = just_cast_it<DeviceViewable>(p_new_value);
		return TRTC_Replace_If(*vec, *pred, *new_value, begin, end);
	}

	bool Native::replace_copy(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_old_value, IntPtr p_new_value, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		DeviceViewable* old_value = just_cast_it<DeviceViewable>(p_old_value);
		DeviceViewable* new_value = just_cast_it<DeviceViewable>(p_new_value);
		return TRTC_Replace_Copy(*vec_in, *vec_out, *old_value, *new_value, begin_in, end_in, begin_out);
	}

	bool Native::replace_copy_if(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_pred, IntPtr p_new_value, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* pred = just_cast_it<Functor>(p_pred);
		DeviceViewable* new_value = just_cast_it<DeviceViewable>(p_new_value);
		return TRTC_Replace_Copy_If(*vec_in, *vec_out, *pred, *new_value, begin_in, end_in, begin_out);
	}

	bool Native::for_each(IntPtr p_vec, IntPtr p_f, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* f = just_cast_it<Functor>(p_f);
		return TRTC_For_Each(*vec, *f, begin, end);
	}

	bool Native::adjacent_difference(IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Adjacent_Difference(*vec_in, *vec_out, begin_in, end_in, begin_out);
	}

	bool Native::adjacent_difference(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_binary_op, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* binary_op = just_cast_it<Functor>(p_binary_op);
		return TRTC_Adjacent_Difference(*vec_in, *vec_out, *binary_op, begin_in, end_in, begin_out);
	}

	bool Native::sequence(IntPtr p_vec, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		return TRTC_Sequence(*vec, begin, end);
	}

	bool Native::sequence(IntPtr p_vec, IntPtr p_value_init, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value_init = just_cast_it<DeviceViewable>(p_value_init);
		return TRTC_Sequence(*vec, *value_init, begin, end);
	}

	bool Native::sequence(IntPtr p_vec, IntPtr p_value_init, IntPtr p_value_step, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value_init = just_cast_it<DeviceViewable>(p_value_init);
		DeviceViewable* value_step = just_cast_it<DeviceViewable>(p_value_step);
		return TRTC_Sequence(*vec, *value_init, *value_step, begin, end);
	}

	bool Native::tabulate(IntPtr p_vec, IntPtr p_op, size_t begin, size_t end)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* op = just_cast_it<Functor>(p_op);
		return TRTC_Tabulate(*vec, *op, begin, end);
	}

	bool Native::transform(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_op, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* op = just_cast_it<Functor>(p_op);
		return TRTC_Transform(*vec_in, *vec_out, *op, begin_in, end_in, begin_out);
	}

	bool Native::transform_binary(IntPtr p_vec_in1, IntPtr p_vec_in2, IntPtr p_vec_out, IntPtr p_op, size_t begin_in1, size_t end_in1, size_t begin_in2, size_t begin_out)
	{
		DVVectorLike* vec_in1 = just_cast_it<DVVectorLike>(p_vec_in1);
		DVVectorLike* vec_in2 = just_cast_it<DVVectorLike>(p_vec_in2);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* op = just_cast_it<Functor>(p_op);
		return TRTC_Transform_Binary(*vec_in1, *vec_in2, *vec_out, *op, begin_in1, end_in1, begin_in2, begin_out);
	}

	bool Native::transform_if(IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_op, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* op = just_cast_it<Functor>(p_op);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Transform_If(*vec_in, *vec_out, *op, *pred, begin_in, end_in, begin_out);
	}

	bool Native::transform_if_stencil(IntPtr p_vec_in, IntPtr p_vec_stencil, IntPtr p_vec_out, IntPtr p_op, IntPtr p_pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_vec_stencil);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* op = just_cast_it<Functor>(p_op);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Transform_If_Stencil(*vec_in, *vec_stencil, *vec_out, *op, *pred, begin_in, end_in, begin_stencil, begin_out);
	}

	bool Native::transform_binary_if_stencil(IntPtr p_vec_in1, IntPtr p_vec_in2, IntPtr p_vec_stencil, IntPtr p_vec_out, IntPtr p_op, IntPtr p_pred, size_t begin_in1, size_t end_in1, size_t begin_in2, size_t begin_stencil, size_t begin_out)
	{
		DVVectorLike* vec_in1 = just_cast_it<DVVectorLike>(p_vec_in1);
		DVVectorLike* vec_in2 = just_cast_it<DVVectorLike>(p_vec_in2);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_vec_stencil);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* op = just_cast_it<Functor>(p_op);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Transform_Binary_If_Stencil(*vec_in1, *vec_in2, *vec_stencil, *vec_out, *op, *pred, begin_in1, end_in1, begin_in2, begin_stencil, begin_out);
	}

}

