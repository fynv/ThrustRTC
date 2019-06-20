#include "copy.h"

bool TRTC_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out)
{
	static TRTC_For s_for(
		{ "view_vec_in", "view_vec_out"}, "idx",
		"    view_vec_out[idx]=(decltype(view_vec_out)::value_t)view_vec_in[idx];\n"
	);

	const DeviceViewable* args[] = { &vec_in, &vec_out };
	return s_for.launch_n(vec_in.size(), args);
}

#include "general_copy_if.h"

uint32_t TRTC_Copy_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred)
{
	Functor src_scan({ {"vec_in", &vec_in}, {"pred", &pred}}, { "idx" },
		"        return pred(vec_in[idx])? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(vec_in.size(), src_scan, vec_in, vec_out);
}

uint32_t TRTC_Copy_If_Stencil(const DVVectorLike& vec_in, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& pred)
{
	Functor src_scan({ {"vec_stencil", &vec_stencil}, {"pred", &pred}}, { "idx" },
		"        return pred(vec_stencil[idx])? (uint32_t)1:(uint32_t)0;");
	return general_copy_if(vec_in.size(), src_scan, vec_in, vec_out);
}
