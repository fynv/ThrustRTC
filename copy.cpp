#include "copy.h"

bool TRTC_Copy(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for(
		{ "view_vec_in", "view_vec_out",  "begin_in", "begin_out" }, "idx",
		"    view_vec_out[idx + begin_out]=(decltype(view_vec_out)::value_t)view_vec_in[idx + begin_in];\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &dvbegin_in, &dvbegin_out };
	return s_for.launch_n(ctx, end_in - begin_in, args);
}

#include "cuda_wrapper.h"
#include "general_scan.h"

uint32_t TRTC_Copy_If(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred, size_t begin_in , size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	DVVector inds(ctx, "uint32_t", n);
	Functor src_scan(ctx, { {"vec_in", &vec_in}, {"pred", &pred}, {"begin_in", &dvbegin_in} }, { "idx" },
		"        return pred(vec_in[idx+begin_in])? (uint32_t)1:(uint32_t)0;\n");
	Functor plus(ctx, {}, { "x", "y" }, "        return x+y;\n");
	if (!general_scan(ctx, n, src_scan, inds, plus, 0)) return (uint32_t)(-1);

	uint32_t ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((uint32_t*)inds.data() + n - 1), sizeof(uint32_t));

	static TRTC_For s_for_scatter({ "vec_in", "inds", "vec_out", "begin_in", "begin_out" }, "idx",
		"    if ((idx==0 && inds[idx]>0) || (idx>0 && inds[idx]>inds[idx-1])) vec_out[inds[idx]-1 + begin_out]=vec_in[idx +begin_in];\n"
	);

	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &inds, &vec_out, &dvbegin_in, &dvbegin_out };
	if (!s_for_scatter.launch_n(ctx, n, args)) return (uint32_t)(-1);
	
	return ret;
}

uint32_t TRTC_Copy_If_Stencil(TRTCContext& ctx, const DVVectorLike& vec_in, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_stencil(begin_stencil);
	DVVector inds(ctx, "uint32_t", n);
	Functor src_scan(ctx, { {"vec_stencil", &vec_stencil}, {"pred", &pred}, {"begin_stencil", &dvbegin_stencil} }, { "idx" },
		"        return pred(vec_stencil[idx+begin_stencil])? (uint32_t)1:(uint32_t)0;");
	Functor plus(ctx, {}, { "x", "y" }, "        return x+y;\n");
	if (!general_scan(ctx, n, src_scan, inds, plus, 0)) return (uint32_t)(-1);

	uint32_t ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((uint32_t*)inds.data() + n - 1), sizeof(uint32_t));

	static TRTC_For s_for_scatter({ "vec_in", "inds", "vec_out", "begin_in", "begin_out" }, "idx",
		"    if ((idx==0 && inds[idx]>0) || (idx>0 && inds[idx]>inds[idx-1])) vec_out[inds[idx]-1 + begin_out]=vec_in[idx +begin_in];\n"
	);

	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &inds, &vec_out, &dvbegin_in, &dvbegin_out };
	if (!s_for_scatter.launch_n(ctx, n, args)) return (uint32_t)(-1);

	return ret;

}