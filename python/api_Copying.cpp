#include "api.h"
#include "TRTCContext.h"
#include "gather.h"
#include "scatter.h"
#include "copy.h"
#include "swap.h"

int n_gather(void* ptr_map, void* ptr_in, void* ptr_out)
{
	DVVectorLike* vec_map = (DVVectorLike*)ptr_map;
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	if (TRTC_Gather(*vec_map, *vec_in, *vec_out))
		return 0;
	else
		return -1;
}

int n_gather_if(void* ptr_map, void* ptr_stencil, void* ptr_in, void* ptr_out, void* ptr_pred)
{
	DVVectorLike* vec_map = (DVVectorLike*)ptr_map;
	DVVectorLike* vec_stencil = (DVVectorLike*)ptr_stencil;
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* pred = (Functor*)ptr_pred;

	if (pred == nullptr)
	{
		if (TRTC_Gather_If(*vec_map, *vec_stencil, *vec_in, *vec_out))
			return 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Gather_If(*vec_map, *vec_stencil, *vec_in, *vec_out, *pred))
			return 0;
		else
			return -1;
	}
}

int n_scatter(void* ptr_in, void* ptr_map, void* ptr_out)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_map = (DVVectorLike*)ptr_map;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	if (TRTC_Scatter(*vec_in, *vec_map, *vec_out))
		return 0;
	else
		return -1;
}

int n_scatter_if(void* ptr_in, void* ptr_map, void* ptr_stencil, void* ptr_out, void* ptr_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_map = (DVVectorLike*)ptr_map;
	DVVectorLike* vec_stencil = (DVVectorLike*)ptr_stencil;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	Functor* pred = (Functor*)ptr_pred;

	if (pred == nullptr)
	{
		if (TRTC_Scatter_If(*vec_in, *vec_map, *vec_stencil, *vec_out))
			return 0;
		else
			return -1;
	}
	else
	{
		if (TRTC_Scatter_If(*vec_in, *vec_map, *vec_stencil, *vec_out, *pred))
			return 0;
		else
			return -1;
	}
}

int n_copy(void* ptr_in, void* ptr_out)
{
	DVVectorLike* vec_in = (DVVectorLike*)ptr_in;
	DVVectorLike* vec_out = (DVVectorLike*)ptr_out;
	if (TRTC_Copy(*vec_in, *vec_out))
		return 0;
	else
		return 1;
}

int n_swap(void* ptr_vec1, void* ptr_vec2)
{
	DVVectorLike* vec1 = (DVVectorLike*)ptr_vec1;
	DVVectorLike* vec2 = (DVVectorLike*)ptr_vec2;
	if (TRTC_Swap(*vec1, *vec2))
		return 0;
	else
		return -1;
}
