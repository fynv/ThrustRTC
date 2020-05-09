#include "api.h"
#include "TRTCContext.h"
#include "copy.h"
#include "remove.h"
#include "unique.h"
#include "partition.h"

unsigned n_copy_if(void* ptr_in, void* ptr_out, void* ptr_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* pred = (Functor*)ptr_pred;
	return TRTC_Copy_If(*vec_in, *vec_out, *pred);
}

unsigned n_copy_if_stencil(void* ptr_in, void* ptr_stencil, void* ptr_out, void* ptr_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_stencil = (DVVectorLike*)ptr_stencil;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* pred = (Functor*)ptr_pred;
	return TRTC_Copy_If_Stencil(*vec_in, *vec_stencil, *vec_out, *pred);
}

unsigned n_remove(void* ptr_vec, void* ptr_value)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DeviceViewable* value = (DeviceViewable*)ptr_value;
	return TRTC_Remove(*vec, *value);
}

unsigned n_remove_copy(void* ptr_in, void* ptr_out, void* ptr_value)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	DeviceViewable* value = (DeviceViewable*)ptr_value;
	return TRTC_Remove_Copy(*vec_in, *vec_out, *value);
}

unsigned n_remove_if(void* ptr_vec, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* pred = (Functor*)ptr_pred;
	return TRTC_Remove_If(*vec, *pred);
}

unsigned n_remove_copy_if(void* ptr_in, void* ptr_out, void* ptr_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* pred = (Functor*)ptr_pred;
	return TRTC_Remove_Copy_If(*vec_in, *vec_out, *pred);
}

unsigned n_remove_if_stencil(void* ptr_vec, void* ptr_stencil, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DVVectorLike* stencil = (DVVectorLike*)ptr_stencil;
	Functor* pred = (Functor*)ptr_pred;
	return TRTC_Remove_If_Stencil(*vec, *stencil, *pred);
}

unsigned n_remove_copy_if_stencil(void* ptr_in, void* ptr_stencil, void* ptr_out, void* ptr_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* stencil = (DVVectorLike*)ptr_stencil;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* pred = (Functor*)ptr_pred;
	return TRTC_Remove_Copy_If_Stencil(*vec_in, *stencil, *vec_out, *pred);
}

unsigned n_unique(void* ptr_vec, void* ptr_binary_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* binary_pred = (Functor*)ptr_binary_pred;
	if (binary_pred == nullptr)
	{
		return TRTC_Unique(*vec);
	}
	else
	{
		return TRTC_Unique(*vec, *binary_pred);
	}
}

unsigned n_unique_copy(void* ptr_vec_in, void* ptr_vec_out, void* ptr_binary_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_vec_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_vec_out;
	Functor* binary_pred = (Functor*)ptr_binary_pred;
	if (binary_pred == nullptr)
	{
		return TRTC_Unique_Copy(*vec_in, *vec_out);
	}
	else
	{
		return TRTC_Unique_Copy(*vec_in, *vec_out, *binary_pred);
	}
}

unsigned n_unique_by_key(void* ptr_keys, void* ptr_values, void* ptr_binary_pred)
{
	DVVectorLike* keys = (DVVectorLike*)ptr_keys;
	DVVectorLike* values = (DVVectorLike*)ptr_values;
	Functor* binary_pred = (Functor*)ptr_binary_pred;
	if (binary_pred == nullptr)
	{
		return TRTC_Unique_By_Key(*keys, *values);
	}
	else
	{
		return TRTC_Unique_By_Key(*keys, *values, *binary_pred);
	}
}

unsigned n_unique_by_key_copy(void* ptr_keys_in, void* ptr_values_in, void* ptr_key_out, void* ptr_values_out, void* ptr_binary_pred)
{
	DVVectorLike* keys_in = (DVVectorLike*)ptr_keys_in;
	DVVectorLike* values_in = (DVVectorLike*)ptr_values_in;
	DVVectorLike* keys_out = (DVVectorLike*)ptr_values_in;
	DVVectorLike* values_out = (DVVectorLike*)ptr_values_out;
	Functor* binary_pred = (Functor*)ptr_binary_pred;
	if (binary_pred == nullptr)
	{
		return TRTC_Unique_By_Key_Copy(*keys_in, *values_in, *keys_out, *values_out);
	}
	else
	{
		return TRTC_Unique_By_Key_Copy(*keys_in, *values_in, *keys_out, *values_out, *binary_pred);
	}
}

unsigned n_partition(void* ptr_vec, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* pred = (Functor*)ptr_pred;
	return TRTC_Partition(*vec, *pred);
}

unsigned n_partition_stencil(void* ptr_vec, void* ptr_stencil, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DVVectorLike* stencil = (DVVectorLike*)ptr_stencil;
	Functor* pred = (Functor*)ptr_pred;
	return TRTC_Partition_Stencil(*vec, *stencil, *pred);
}

unsigned n_partition_copy(void* ptr_vec_in, void* ptr_vec_true, void* ptr_vec_false, void* ptr_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_vec_in;
	DVVectorLike* vec_true = (DVVectorLike*)ptr_vec_true;
	DVVectorLike* vec_false = (DVVectorLike*)ptr_vec_false;
	Functor* pred = (Functor*)ptr_pred;
	return TRTC_Partition_Copy(*vec_in, *vec_true, *vec_false, *pred);
}

unsigned n_partition_copy_stencil(void* ptr_vec_in, void* ptr_stencil, void* ptr_vec_true, void* ptr_vec_false, void* ptr_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_vec_in;
	DVVectorLike* stencil = (DVVectorLike*)ptr_stencil;
	DVVectorLike* vec_true = (DVVectorLike*)ptr_vec_true;
	DVVectorLike* vec_false = (DVVectorLike*)ptr_vec_false;
	Functor* pred = (Functor*)ptr_pred;
	return TRTC_Partition_Copy_Stencil(*vec_in, *stencil, *vec_true, *vec_false, *pred);
}
