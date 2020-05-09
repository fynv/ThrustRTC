#include "api.h"
#include "TRTCContext.h"
#include "fill.h"
#include "replace.h"
#include "for_each.h"
#include "adjacent_difference.h"
#include "sequence.h"
#include "tabulate.h"
#include "transform.h"

int n_fill(void* ptr_vec, void* ptr_value)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DeviceViewable* value = (DeviceViewable*)ptr_value;
	if (TRTC_Fill(*vec, *value))
		return 0;
	else
		return -1;
}

int n_replace(void* ptr_vec, void* ptr_old_value, void* ptr_new_value)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DeviceViewable* old_value = (DeviceViewable*)ptr_old_value;
	DeviceViewable* new_value = (DeviceViewable*)ptr_new_value;
	if (TRTC_Replace(*vec, *old_value, *new_value))
		return 0;
	else
		return -1;
}

int n_replace_if(void* ptr_vec, void* p_pred, void* ptr_new_value)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* pred = (Functor*)p_pred;
	DeviceViewable* new_value = (DeviceViewable*)ptr_new_value;
	if (TRTC_Replace_If(*vec, *pred, *new_value))
		return 0;
	else
		return -1;
}

int n_replace_copy(void* ptr_in, void* ptr_out, void* ptr_old_value, void* ptr_new_value)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	DeviceViewable* old_value = (DeviceViewable*)ptr_old_value;
	DeviceViewable* new_value = (DeviceViewable*)ptr_new_value;
	if (TRTC_Replace_Copy(*vec_in, *vec_out, *old_value, *new_value))
		return 0;
	else
		return -1;
}

int n_replace_copy_if(void* ptr_in, void* ptr_out, void* p_pred, void* ptr_new_value)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* pred = (Functor*)p_pred;
	DeviceViewable* new_value = (DeviceViewable*)ptr_new_value;
	if (TRTC_Replace_Copy_If(*vec_in, *vec_out, *pred, *new_value))
		return 0;
	else
		return -1;
}

int n_for_each(void* ptr_vec, void* ptr_f)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* f = (Functor*)ptr_f;
	if (TRTC_For_Each(*vec, *f))
		return 0;
	else
		return -1;
}

int n_adjacent_difference(void* ptr_in, void* ptr_out, void* ptr_binary_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* binary_op = (Functor*)ptr_binary_op;
	
	if (binary_op == nullptr)
	{
		if (TRTC_Adjacent_Difference(*vec_in, *vec_out))
			return 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Adjacent_Difference(*vec_in, *vec_out, *binary_op))
			return 0;
		else
			return -1;
	}
}

int n_sequence(void* ptr_vec, void* ptr_value_init, void* ptr_value_step)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DeviceViewable* value_init = (DeviceViewable*)ptr_value_init;
	DeviceViewable* value_step = (DeviceViewable*)ptr_value_step;
	if (value_init == nullptr)
	{
		if (TRTC_Sequence(*vec))
			return 0;
		else
			return -1;
	}
	else if (value_step == nullptr)
	{
		if (TRTC_Sequence(*vec, *value_init))
			return 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Sequence(*vec, *value_init, *value_step))
			return 0;
		else
			return -1;
	}
}

int n_tabulate(void* ptr_vec, void* ptr_op)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* op = (Functor*)ptr_op;
	if (TRTC_Tabulate(*vec, *op))
		return 0;
	else
		return -1;
}


int n_transform(void* ptr_in, void* ptr_out, void* ptr_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* op = (Functor*)ptr_op;
	if (TRTC_Transform(*vec_in, *vec_out, *op))
		return 0;
	else
		return -1;
}

int n_transform_binary(void* ptr_in1, void* ptr_in2, void* ptr_out, void* ptr_op)
{
	DVVectorLike* vec_in1 = (DVVectorLike*)ptr_in1;
	DVVectorLike* vec_in2 = (DVVectorLike*)ptr_in2;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* op = (Functor*)ptr_op;
	if (TRTC_Transform_Binary(*vec_in1, *vec_in2, *vec_out, *op))
		return 0;
	else
		return -1;
}

int n_transform_if(void* ptr_in, void* ptr_out, void* ptr_op, void* ptr_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* op = (Functor*)ptr_op;
	Functor* pred = (Functor*)ptr_pred;
	TRTC_Transform_If(*vec_in, *vec_out, *op, *pred);
	return 0;
}

int n_transform_if_stencil(void* ptr_in, void* ptr_stencil, void* ptr_out, void* ptr_op, void* ptr_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_stencil = (DVVectorLike*)ptr_stencil;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* op = (Functor*)ptr_op;
	Functor* pred = (Functor*)ptr_pred;
	if (TRTC_Transform_If_Stencil(*vec_in, *vec_stencil, *vec_out, *op, *pred))
		return 0;
	else
		return -1;
}

int n_transform_binary_if_stencil(void* ptr_in1, void* ptr_in2, void* ptr_stencil, void* ptr_out, void* ptr_op, void* ptr_pred)
{
	DVVectorLike* vec_in1 = (DVVectorLike*)ptr_in1;
	DVVectorLike* vec_in2 = (DVVectorLike*)ptr_in2;
	DVVectorLike* vec_stencil = (DVVectorLike*)ptr_stencil;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* op = (Functor*)ptr_op;
	Functor* pred = (Functor*)ptr_pred;
	if (TRTC_Transform_Binary_If_Stencil(*vec_in1, *vec_in2, *vec_stencil, *vec_out, *op, *pred))
		return 0;
	else
		return -1;
}




