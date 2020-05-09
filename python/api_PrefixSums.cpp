#include "api.h"
#include "TRTCContext.h"
#include "scan.h"
#include "transform_scan.h"

int n_inclusive_scan(void* ptr_vec_in, void* ptr_vec_out, void* ptr_binary_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_vec_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_vec_out;
	Functor* binary_op = (Functor*)ptr_binary_op;

	if (binary_op == nullptr)
	{
		if (!TRTC_Inclusive_Scan(*vec_in, *vec_out)) return -1;
	}
	else
	{
		if (!TRTC_Inclusive_Scan(*vec_in, *vec_out, *binary_op)) return -1;
	}
	return 0;
}

int n_exclusive_scan(void* ptr_vec_in, void* ptr_vec_out, void* ptr_init, void* ptr_binary_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_vec_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_vec_out;
	DeviceViewable* init = (DeviceViewable*)ptr_init;
	Functor* binary_op = (Functor*)ptr_binary_op;

	if (init == nullptr)
	{
		if (!TRTC_Exclusive_Scan(*vec_in, *vec_out)) return -1;
	}
	else if (binary_op == nullptr)
	{
		if (!TRTC_Exclusive_Scan(*vec_in, *vec_out, *init)) return -1;
	}
	else
	{
		if (!TRTC_Exclusive_Scan(*vec_in, *vec_out, *init, *binary_op)) return -1;
	}
	return 0;
}

int n_inclusive_scan_by_key(void* ptr_vec_key, void* ptr_vec_value, void* ptr_vec_out, void* ptr_binary_pred, void* ptr_binary_op)
{
	DVVectorLike* vec_key = (DVVectorLike*)ptr_vec_key;
	DVVectorLike* vec_value = (DVVectorLike*)ptr_vec_value;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_vec_out;
	Functor* binary_pred = (Functor*)ptr_binary_pred;
	Functor* binary_op = (Functor*)ptr_binary_op;

	if (binary_pred == nullptr)
	{
		if (!TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out)) return -1;
	}
	else if (binary_op == nullptr)
	{
		if (!TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *binary_pred)) return -1;
	}
	else
	{
		if (!TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *binary_pred, *binary_op)) return -1;
	}
	return 0;
}

int n_exclusive_scan_by_key(void* ptr_vec_key, void* ptr_vec_value, void* ptr_vec_out, void* ptr_init, void* ptr_binary_pred, void* ptr_binary_op)
{
	DVVectorLike* vec_key = (DVVectorLike*)ptr_vec_key;
	DVVectorLike* vec_value = (DVVectorLike*)ptr_vec_value;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_vec_out;
	DeviceViewable* init = (DeviceViewable*)ptr_init;
	Functor* binary_pred = (Functor*)ptr_binary_pred;
	Functor* binary_op = (Functor*)ptr_binary_op;

	if (init == nullptr)
	{
		if (!TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out)) return -1;
	}
	else if (binary_pred == nullptr)
	{
		if (!TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init)) return -1;
	}
	else if (binary_op == nullptr)
	{
		if (!TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init, *binary_pred)) return -1;
	}
	else
	{
		if (!TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init, *binary_pred, *binary_op)) return -1;
	}
	return 0;
}

int n_transform_inclusive_scan(void* ptr_vec_in, void* ptr_vec_out, void* ptr_unary_op, void* ptr_binary_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_vec_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_vec_out;
	Functor* unary_op = (Functor*)ptr_unary_op;
	Functor* binary_op = (Functor*)ptr_binary_op;
	return TRTC_Transform_Inclusive_Scan(*vec_in, *vec_out, *unary_op, *binary_op) ? 0 : -1;
}

int n_transform_exclusive_scan(void* ptr_vec_in, void* ptr_vec_out, void* ptr_unary_op, void* ptr_init, void* ptr_binary_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_vec_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_vec_out;
	Functor* unary_op = (Functor*)ptr_unary_op;
	DeviceViewable* init = (DeviceViewable*)ptr_init;
	Functor* binary_op = (Functor*)ptr_binary_op;
	return TRTC_Transform_Exclusive_Scan(*vec_in, *vec_out, *unary_op, *init, *binary_op) ? 0 : 1;
}
