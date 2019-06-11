#include "copy.h"

bool TRTC_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for(
		{ "view_vec_in", "view_vec_out",  "begin_in", "begin_out" }, "idx",
		"    view_vec_out[idx + begin_out]=(decltype(view_vec_out)::value_t)view_vec_in[idx + begin_in];\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &dvbegin_in, &dvbegin_out };
	return s_for.launch_n(end_in - begin_in, args);
}

#include "general_copy_if.h"

uint32_t TRTC_Copy_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred, size_t begin_in , size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src_scan({ {"vec_in", &vec_in}, {"pred", &pred}, {"begin_in", &dvbegin_in} }, { "idx" },
		"        return pred(vec_in[idx+begin_in])? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(n, src_scan, vec_in, vec_out, begin_in, begin_out);
}

uint32_t TRTC_Copy_If_Stencil(const DVVectorLike& vec_in, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_stencil(begin_stencil);
	Functor src_scan({ {"vec_stencil", &vec_stencil}, {"pred", &pred}, {"begin_stencil", &dvbegin_stencil} }, { "idx" },
		"        return pred(vec_stencil[idx+begin_stencil])? (uint32_t)1:(uint32_t)0;");
	return general_copy_if(n, src_scan, vec_in, vec_out, begin_in, begin_out);
}
