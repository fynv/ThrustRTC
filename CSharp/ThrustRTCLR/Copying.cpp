
#include "ThrustRTCLR.h"

#include "gather.h"
#include "scatter.h"
#include "copy.h"
#include "swap.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	bool Native::gather(IntPtr p_vec_map, IntPtr p_vec_in, IntPtr p_vec_out)
	{
		DVVectorLike* vec_map = just_cast_it<DVVectorLike>(p_vec_map);
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Gather(*vec_map, *vec_in, *vec_out);
	}

	bool Native::gather_if(IntPtr p_vec_map, IntPtr p_vec_stencil, IntPtr p_vec_in, IntPtr p_vec_out)
	{
		DVVectorLike* vec_map = just_cast_it<DVVectorLike>(p_vec_map);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_vec_stencil);
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Gather_If(*vec_map, *vec_stencil, *vec_in, *vec_out);
	}

	bool Native::gather_if(IntPtr p_vec_map, IntPtr p_vec_stencil, IntPtr p_vec_in, IntPtr p_vec_out, IntPtr p_pred)
	{
		DVVectorLike* vec_map = just_cast_it<DVVectorLike>(p_vec_map);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_vec_stencil);
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Gather_If(*vec_map, *vec_stencil, *vec_in, *vec_out, *pred);
	}

	bool Native::scatter(IntPtr p_vec_in, IntPtr p_vec_map, IntPtr p_vec_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_map = just_cast_it<DVVectorLike>(p_vec_map);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Scatter(*vec_in, *vec_map, *vec_out);
	}

	bool Native::scatter_if(IntPtr p_vec_in, IntPtr p_vec_map, IntPtr p_vec_stencil, IntPtr p_vec_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_map = just_cast_it<DVVectorLike>(p_vec_map);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_vec_stencil);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Scatter_If(*vec_in, *vec_map, *vec_stencil, *vec_out);
	}

	bool Native::scatter_if(IntPtr p_vec_in, IntPtr p_vec_map, IntPtr p_vec_stencil, IntPtr p_vec_out, IntPtr p_pred)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_map = just_cast_it<DVVectorLike>(p_vec_map);
		DVVectorLike* vec_stencil = just_cast_it<DVVectorLike>(p_vec_stencil);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		Functor* pred = just_cast_it<Functor>(p_pred);
		return TRTC_Scatter_If(*vec_in, *vec_map, *vec_stencil, *vec_out, *pred);
	}

	bool Native::copy(IntPtr p_vec_in, IntPtr p_vec_out)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		DVVectorLike* vec_out = just_cast_it<DVVectorLike>(p_vec_out);
		return TRTC_Copy(*vec_in, *vec_out);
	}
	
	bool Native::swap(IntPtr p_vec1, IntPtr p_vec2)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		return TRTC_Swap(*vec1, *vec2);
	}
}