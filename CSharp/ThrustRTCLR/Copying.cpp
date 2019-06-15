#include "stdafx.h"
#include "ThrustRTCLR.h"

#include "gather.h"
#include "copy.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	bool Native::gather(IntPtr p_vec_map, IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_map, size_t end_map, size_t begin_in, size_t begin_out)
	{
		DVVectorLike* vec_map = just_cast_it<DVVectorLike>(p_vec_map);
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Gather(*vec_map, *vec_in, *vec_out, begin_map, end_map, begin_in, begin_out);
	}

	bool Native::gather_if(IntPtr p_vec_map, IntPtr p_vec_stencil, IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_map, size_t end_map, size_t begin_stencil, size_t begin_in, size_t begin_out)
	{
		DVVectorLike* vec_map = just_cast_it<DVVectorLike>(p_vec_map);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_vec_stencil);
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Gather_If(*vec_map, *vec_stencil, *vec_in, *vec_out, begin_map, end_map, begin_stencil, begin_in, begin_out);
	}

	bool Native::gather_if(IntPtr p_vec_map, IntPtr p_vec_stencil, IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_pred, size_t begin_map, size_t end_map, size_t begin_stencil, size_t begin_in, size_t begin_out)
	{
		DVVectorLike* vec_map = just_cast_it<DVVectorLike>(p_vec_map);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_vec_stencil);
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Gather_If(*vec_map, *vec_stencil, *vec_in, *vec_out, *pred, begin_map, end_map, begin_stencil, begin_in, begin_out);
	}

	bool Native::copy(IntPtr p_vec_in, IntPtr p_vec_out, size_t begin_in, size_t end_in, size_t begin_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Copy(*vec_in, *vec_out, begin_in, end_in, begin_out);
	}
}