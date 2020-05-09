#include "api.h"
#include "TRTCContext.h"
#include "sort.h"

int n_sort(void* ptr_vec, void* ptr_comp)
{
	DVVectorLike* vec = (DVVectorLike*)ptr_vec;
	Functor* comp = (Functor*)ptr_comp;

	if (comp == nullptr)
	{
		if (TRTC_Sort(*vec))
			return 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Sort(*vec, *comp))
			return 0;
		else
			return -1;
	}
}

int n_sort_by_key(void* ptr_keys, void* ptr_values, void* ptr_comp)
{
	DVVectorLike* keys = (DVVectorLike*)ptr_keys;
	DVVectorLike* values = (DVVectorLike*)ptr_values;
	Functor* comp = (Functor*)ptr_comp;

	if (comp == nullptr)
	{
		if (TRTC_Sort_By_Key(*keys, *values))
			return 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Sort_By_Key(*keys, *values, *comp))
			return 0;
		else
			return -1;
	}
}

