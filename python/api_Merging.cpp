#include "api.h"
#include "TRTCContext.h"
#include "merge.h"

int n_merge(void* ptr_vec1, void* ptr_vec2, void* ptr_vec_out, void* ptr_comp)
{
	DVVectorLike* vec1 = (DVVectorLike*)ptr_vec1;
	DVVectorLike* vec2 = (DVVectorLike*)ptr_vec2;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_vec_out;
	Functor* comp = (Functor*)ptr_comp;

	if (comp == nullptr)
	{
		if (TRTC_Merge(*vec1, *vec2, *vec_out))
			return 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Merge(*vec1, *vec2, *vec_out, *comp))
			return 0;
		else
			return -1;
	}
}

int n_merge_by_key(void* ptr_keys1, void* ptr_keys2, void* ptr_value1, void* ptr_value2, void* ptr_keys_out, void* ptr_value_out, void* ptr_comp)
{

	DVVectorLike* keys1 = (DVVectorLike*)ptr_keys1;
	DVVectorLike* keys2 = (DVVectorLike*)ptr_keys2;
	DVVectorLike* value1 = (DVVectorLike*)ptr_value1;
	DVVectorLike* value2 = (DVVectorLike*)ptr_value2;
	DVVectorLike* keys_out = (DVVectorLike*)ptr_keys_out;
	DVVectorLike* value_out = (DVVectorLike*)ptr_value_out;
	Functor* comp = (Functor*)ptr_comp;

	if (comp == nullptr)
	{
		if (TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out))
			return 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out, *comp))
			return 0;
		else
			return -1;
	}

}
