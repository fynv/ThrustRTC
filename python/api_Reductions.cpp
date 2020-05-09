#include "api.h"
#include "TRTCContext.h"
#include "count.h"
#include "reduce.h"
#include "equal.h"
#include "extrema.h"
#include "inner_product.h"
#include "transform_reduce.h"
#include "logical.h"
#include "partition.h"
#include "sort.h"

unsigned long long n_count(void* ptr_vec, void* ptr_value)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DeviceViewable* value = (DeviceViewable*)ptr_value;
	size_t res;
	if (TRTC_Count(*vec, *value, res))
		return (unsigned long long)res;
	else
		return (unsigned long long)(-1);

}

unsigned long long n_count_if(void* ptr_vec, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* pred = (Functor*)ptr_pred;
	size_t res;
	if (TRTC_Count_If(*vec, *pred, res))
		return (unsigned long long)res;
	else
		return (unsigned long long)(-1);
}

void* n_reduce(void* ptr_vec, void* ptr_init, void* ptr_bin_op)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DeviceViewable* init = (DeviceViewable*)ptr_init;
	Functor* binary_op = (Functor*)ptr_bin_op;
	
	ViewBuf ret;
	if (init == nullptr)
	{
		if (!TRTC_Reduce(*vec, ret))
			return nullptr;
	}
	else if (binary_op == nullptr)
	{
		if (!TRTC_Reduce(*vec, *init, ret))
			return nullptr;
	}
	else
	{
		if (!TRTC_Reduce(*vec, *init, *binary_op, ret))
			return nullptr;
	}
	return dv_from_viewbuf(ret, vec->name_elem_cls().c_str());
}

unsigned n_reduce_by_key(void* ptr_key_in, void* ptr_value_in, void* ptr_key_out, void* ptr_value_out, void* ptr_binary_pred, void* ptr_binary_op)
{
	DVVectorLike* key_in = (DVVectorLike*)ptr_key_in;
	DVVectorLike* value_in = (DVVectorLike*)ptr_value_in;
	DVVectorLike* key_out = (DVVectorLike*)ptr_key_out;
	DVVectorLike* value_out = (DVVectorLike*)ptr_value_out;
	Functor* binary_pred = (Functor*)ptr_binary_pred;;
	Functor* binary_op = (Functor*)ptr_binary_op;

	if (binary_pred == nullptr)
	{
		return TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out);	
	}
	else if (binary_op == nullptr)
	{
		return TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out, *binary_pred);
	}
	else
	{
		return TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out, *binary_pred, *binary_op);
	}
}

int n_equal(void* ptr_vec1, void* ptr_vec2, void* ptr_binary_pred)
{
	DVVectorLike* vec1 = (DVVectorLike*)ptr_vec1;
	DVVectorLike* vec2 = (DVVectorLike*)ptr_vec2;
	Functor* binary_pred = (Functor*)ptr_binary_pred;

	bool res;
	if (binary_pred == nullptr)
		if (TRTC_Equal(*vec1, *vec2, res))
			return res ? 1 : 0;
		else
			return -1;
	else
		if (TRTC_Equal(*vec1, *vec2, *binary_pred, res))
			return res ? 1 : 0;
		else
			return -1;
}

unsigned long long n_min_element(void* ptr_vec, void* ptr_comp)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* comp = (Functor*)ptr_comp;

	size_t id_min;
	if (comp == nullptr)
		if (TRTC_Min_Element(*vec, id_min))
			return (unsigned long long)id_min;
		else
			return (unsigned long long)(-1);
	else
		if (TRTC_Min_Element(*vec, *comp, id_min))
			return (unsigned long long)id_min;
		else
			return (unsigned long long)(-1);
}


unsigned long long n_max_element(void* ptr_vec, void* ptr_comp)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* comp = (Functor*)ptr_comp;

	size_t id_max;
	if (comp == nullptr)
		if (TRTC_Max_Element(*vec, id_max))
			return (unsigned long long)id_max;
		else
			return (unsigned long long)(-1);
	else
		if (TRTC_Max_Element(*vec, *comp, id_max))
			return (unsigned long long)id_max;
		else
			return (unsigned long long)(-1);
}


int n_minmax_element(void* ptr_vec, void* ptr_comp, unsigned long long* ret)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* comp = (Functor*)ptr_comp;
	
	size_t id_min, id_max;
	if (comp == nullptr)
	{
		if (!TRTC_MinMax_Element(*vec, id_min, id_max)) return -1;
	}
	else
	{
		if (!TRTC_MinMax_Element(*vec, *comp, id_min, id_max)) return -1;
	}

	ret[0] = id_min;
	ret[1] = id_max;	
	return 0;
}

void* n_inner_product(void* ptr_vec1, void* ptr_vec2, void* ptr_init, void* ptr_binary_op1, void* ptr_binary_op2)
{
	DVVectorLike* vec1 = (DVVectorLike*)ptr_vec1;
	DVVectorLike* vec2 = (DVVectorLike*)ptr_vec2;
	DeviceViewable* init = (DeviceViewable*)ptr_init;

	Functor* binary_op1 = (Functor*)ptr_binary_op1;
	Functor* binary_op2 = (Functor*)ptr_binary_op2;

	ViewBuf ret;
	if (binary_op1==nullptr || binary_op2 == nullptr)
	{
		if (!TRTC_Inner_Product(*vec1, *vec2, *init, ret)) return nullptr;
	}
	else
	{
		if (!TRTC_Inner_Product(*vec1, *vec2, *init, ret, *binary_op1, *binary_op2)) return nullptr;
	}
	return dv_from_viewbuf(ret, init->name_view_cls().c_str());
}

void* n_transform_reduce(void* ptr_vec, void* ptr_unary_op, void* ptr_init, void* ptr_binary_op)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* unary_op = (Functor*)ptr_unary_op;
	DeviceViewable* init = (DeviceViewable*)ptr_init;
	Functor* binary_op = (Functor*)ptr_binary_op;
	ViewBuf ret;
	if (!TRTC_Transform_Reduce(*vec, *unary_op, *init, *binary_op, ret)) return nullptr;
	return dv_from_viewbuf(ret, init->name_view_cls().c_str());
}

int n_all_of(void* ptr_vec, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* pred = (Functor*)ptr_pred;
	bool res;
	if (TRTC_All_Of(*vec, *pred, res))
		return res ? 1 : 0;
	else
		return -1;
}

int n_any_of(void* ptr_vec, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* pred = (Functor*)ptr_pred;
	bool res;
	if (TRTC_Any_Of(*vec, *pred, res))
		return res ? 1 : 0;
	else
		return -1;
}

int n_none_of(void* ptr_vec, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* pred = (Functor*)ptr_pred;
	bool res;
	if (TRTC_None_Of(*vec, *pred, res))
		return res ? 1 : 0;
	else
		return -1;
}

int n_is_partitioned(void* ptr_vec, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* pred = (Functor*)ptr_pred;
	bool res;
	if (TRTC_Is_Partitioned(*vec, *pred, res))
		return res ? 1 : 0;
	else
		return -1;
}

int n_is_sorted(void* ptr_vec, void* ptr_comp)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* comp = (Functor*)ptr_comp;
	bool res;
	if (comp == nullptr)
	{
		if (TRTC_Is_Sorted(*vec, res))
			return res ? 1 : 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Is_Sorted(*vec, *comp, res))
			return res ? 1 : 0;
		else
			return -1;
	}
}
