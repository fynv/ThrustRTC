#include "api.h"
#include "TRTCContext.h"
#include "find.h"
#include "mismatch.h"
#include "binary_search.h"
#include "partition.h"
#include "sort.h"

long long n_find(void* ptr_vec, void* ptr_value)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DeviceViewable* value = (DeviceViewable*)ptr_value;
	size_t res;
	if (TRTC_Find(*vec, *value, res))
	{
		if (res == (size_t)(-1))
			return (long long)(vec->size());
		return (long long)res;
	}
	else
	{
		return -1;
	}
}

long long n_find_if(void* ptr_vec, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* pred = (Functor*)ptr_pred;
	size_t res;
	if (TRTC_Find_If(*vec, *pred, res))
	{
		if (res == (size_t)(-1))
			return (long long)(vec->size());
		return (long long)res;
	}
	else
	{
		return -1;
	}
}

long long n_find_if_not(void* ptr_vec, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* pred = (Functor*)ptr_pred;
	size_t res;
	if (TRTC_Find_If_Not(*vec, *pred, res))
	{
		if (res == (size_t)(-1))
			return (long long)(vec->size());
		return (long long)res;
	}
	else
	{
		return -1;
	}
}

long long n_mismatch(void* ptr_vec1, void* ptr_vec2, void* ptr_pred)
{
	DVVectorLike* vec1 = (DVVectorLike*)ptr_vec1;
	DVVectorLike* vec2 = (DVVectorLike*)ptr_vec2;
	Functor* pred = (Functor*)(DVVectorLike*)ptr_pred;
	size_t res;
	if (pred == nullptr)
	{
		if (TRTC_Mismatch(*vec1, *vec2, res))
		{
			if (res == (size_t)(-1))
				return (long long)(vec1->size());
			return (long long)res;
		}
		else
		{
			return -1;
		}
	}
	else
	{
		if (TRTC_Mismatch(*vec1, *vec2, *pred, res))
		{
			if (res == (size_t)(-1))
				return (long long)(vec1->size());
			return (long long)res;
		}
		else
		{
			return -1;
		}
	}
}

unsigned long long n_lower_bound(void* ptr_vec, void* ptr_value, void* ptr_comp)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DeviceViewable* value = (DeviceViewable*)ptr_value;
	Functor* comp = (Functor*)ptr_comp;
	if (comp == nullptr)
	{
		size_t res;
		if (TRTC_Lower_Bound(*vec, *value, res))
		{			
			return res;
		}
		else
		{
			return (unsigned long long)(-1);
		}
	}
	else
	{
		size_t res;
		if (TRTC_Lower_Bound(*vec, *value, *comp, res))
		{
			return res;
		}
		else
		{
			return (unsigned long long)(-1);
		}
	}
}

unsigned long long n_upper_bound(void* ptr_vec, void* ptr_value, void* ptr_comp)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DeviceViewable* value = (DeviceViewable*)ptr_value;
	Functor* comp = (Functor*)ptr_comp;
	if (comp == nullptr)
	{
		size_t res;
		if (TRTC_Upper_Bound(*vec, *value, res))
		{
			return res;
		}
		else
		{
			return (unsigned long long)(-1);
		}
	}
	else
	{
		size_t res;
		if (TRTC_Upper_Bound(*vec, *value, *comp, res))
		{
			return res;
		}
		else
		{
			return (unsigned long long)(-1);
		}
	}
}

int n_binary_search(void* ptr_vec, void* ptr_value, void* ptr_comp)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DeviceViewable* value = (DeviceViewable*)ptr_value;
	Functor* comp = (Functor*)ptr_comp;
	if (comp == nullptr)
	{
		bool res;
		if (TRTC_Binary_Search(*vec, *value, res))
			return res ? 1 : 0;
		else
			return -1;
	}
	else
	{
		bool res;
		if (TRTC_Binary_Search(*vec, *value, *comp, res))
			return res ? 1 : 0;
		else
			return -1;
	}
}

int n_lower_bound_v(void* ptr_vec, void* ptr_values, void* ptr_result, void* ptr_comp)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DVVectorLike* values = (DVVectorLike*)ptr_values;
	DVVectorLike* result = (DVVectorLike*)ptr_result;
	Functor* comp = (Functor*)ptr_comp;

	if (comp == nullptr)
	{
		if (TRTC_Lower_Bound_V(*vec, *values, *result))
			return 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Lower_Bound_V(*vec, *values, *result, *comp))
			return 0;
		else
			return -1;
	}
}

int n_upper_bound_v(void* ptr_vec, void* ptr_values, void* ptr_result, void* ptr_comp)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DVVectorLike* values = (DVVectorLike*)ptr_values;
	DVVectorLike* result = (DVVectorLike*)ptr_result;
	Functor* comp = (Functor*)ptr_comp;

	if (comp == nullptr)
	{
		if (TRTC_Upper_Bound_V(*vec, *values, *result))
			return 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Upper_Bound_V(*vec, *values, *result, *comp))
			return 0;
		else
			return -1;
	}
}

int n_binary_search_v(void* ptr_vec, void* ptr_values, void* ptr_result, void* ptr_comp)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	DVVectorLike* values = (DVVectorLike*)ptr_values;
	DVVectorLike* result = (DVVectorLike*)ptr_result;
	Functor* comp = (Functor*)ptr_comp;
	
	if (comp == nullptr)
	{
		if (TRTC_Binary_Search_V(*vec, *values, *result))
			return 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Binary_Search_V(*vec, *values, *result, *comp))
			return 0;
		else
			return -1;
	}
}

unsigned long long n_partition_point(void* ptr_vec, void* ptr_pred)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* pred = (Functor*)ptr_pred;
	size_t pp;
	if (TRTC_Partition_Point(*vec, *pred, pp))
	{
		if (pp == (size_t)(-1)) return vec->size();
		return pp;
	}
	else
	{
		return (unsigned long long)(-1);
	}
}

unsigned long long n_is_sorted_until(void* ptr_vec, void* ptr_comp)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* comp = (Functor*)ptr_comp;
	size_t res;
	if (comp == nullptr)
	{
		if (TRTC_Is_Sorted_Until(*vec, res))
		{
			return res;
		}
		else
		{
			return (unsigned long long)(-1);
		}
	}
	else
	{
		if (TRTC_Is_Sorted_Until(*vec, *comp, res))
		{
			return res;
		}
		else
		{
			return (unsigned long long)(-1);
		}
	}
}
