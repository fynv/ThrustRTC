#include "cuda_wrapper.h"
#include "general_scan.h"

uint32_t general_copy_if(TRTCContext& ctx, size_t n, const Functor& src_scan, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t begin_out)
{
	DVVector inds(ctx, "uint32_t", n);
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
