#include "copy.h"
#include "partition.h"
#include "general_scan.h"
#include "cuda_wrapper.h"
#include "built_in.h"

uint32_t TRTC_Partition(TRTCContext& ctx, DVVectorLike& vec, const Functor& pred, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	size_t n = end - begin;
	DVVector cpy(ctx, vec.name_elem_cls().c_str(), n);
	TRTC_Copy(ctx, vec, cpy, begin, end);

	Functor src_scan(ctx, { {"src", &cpy}, { "pred", &pred } }, { "idx" },
		"        return pred(src[idx]) ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");
	DVVector inds(ctx, "Pair<uint32_t, uint32_t>", n);
	Functor plus(ctx, {}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");
	if (!general_scan(ctx, n, src_scan, inds, plus, 0)) return (uint32_t)(-1);
	Pair<uint32_t, uint32_t> ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + n - 1), sizeof(Pair<uint32_t, uint32_t>));

	static TRTC_For s_for_scatter({ "vec_cpy", "inds", "vec", "begin", "count" }, "idx",
		"    if ((idx==0 && inds[idx].first>0) || (idx>0 && inds[idx].first>inds[idx-1].first))\n"
		"        vec[inds[idx].first -1 + begin] = vec_cpy[idx];\n"
		"    else\n"
		"        vec[count + inds[idx].second - 1 + begin] = vec_cpy[idx];\n"
	);
	DVUInt32 dvcount(ret.first);
	DVSizeT dvbegin(begin);
	const DeviceViewable* args[] = { &cpy, &inds, &vec, &dvbegin, &dvcount };
	if (!s_for_scatter.launch_n(ctx, n, args)) return (uint32_t)(-1);
	return ret.first;
}

uint32_t TRTC_Partition_Stencil(TRTCContext& ctx, DVVectorLike& vec, const DVVectorLike& stencil, const Functor& pred, size_t begin, size_t end, size_t begin_stencil)
{
	if (end == (size_t)(-1)) end = vec.size();
	size_t n = end - begin;
	DVVector cpy(ctx, vec.name_elem_cls().c_str(), n);
	TRTC_Copy(ctx, vec, cpy, begin, end);

	DVSizeT dvbegin_stencil(begin_stencil);
	Functor src_scan(ctx, { {"stencil", &stencil}, {"begin_stencil", &dvbegin_stencil}, { "pred", &pred } }, { "idx" },
		"        return pred(stencil[idx + begin_stencil]) ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");
	DVVector inds(ctx, "Pair<uint32_t, uint32_t>", n);
	Functor plus(ctx, {}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");
	if (!general_scan(ctx, n, src_scan, inds, plus, 0)) return (uint32_t)(-1);
	Pair<uint32_t, uint32_t> ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + n - 1), sizeof(Pair<uint32_t, uint32_t>));

	static TRTC_For s_for_scatter({ "vec_cpy", "inds", "vec", "begin", "count" }, "idx",
		"    if ((idx==0 && inds[idx].first>0) || (idx>0 && inds[idx].first>inds[idx-1].first))\n"
		"        vec[inds[idx].first -1 + begin] = vec_cpy[idx];\n"
		"    else\n"
		"        vec[count + inds[idx].second - 1 + begin] = vec_cpy[idx];\n"
	);
	DVUInt32 dvcount(ret.first);
	DVSizeT dvbegin(begin);
	const DeviceViewable* args[] = { &cpy, &inds, &vec, &dvbegin, &dvcount };
	if (!s_for_scatter.launch_n(ctx, n, args)) return (uint32_t)(-1);
	return ret.first;
}

uint32_t TRTC_Partition_Copy(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_true, DVVectorLike& vec_false, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_true, size_t begin_false)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src_scan(ctx, { {"src", &vec_in},{"begin_in", &dvbegin_in},  { "pred", &pred } }, { "idx" },
		"        return pred(src[idx]) ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");
	DVVector inds(ctx, "Pair<uint32_t, uint32_t>", n);
	Functor plus(ctx, {}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");
	if (!general_scan(ctx, n, src_scan, inds, plus, 0)) return (uint32_t)(-1);
	Pair<uint32_t, uint32_t> ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + n - 1), sizeof(Pair<uint32_t, uint32_t>));

	static TRTC_For s_for_scatter({ "vec_in", "inds", "vec_true", "vec_false", "begin_in", "begin_true", "begin_false" }, "idx",
		"    if ((idx==0 && inds[idx].first>0) || (idx>0 && inds[idx].first>inds[idx-1].first))\n"
		"        vec_true[inds[idx].first -1 + begin_true] = vec_in[idx + begin_in];\n"
		"    else\n"
		"        vec_false[inds[idx].second - 1 + begin_false] = vec_in[idx + begin_in];\n"
	);

	DVSizeT dvbegin_true(begin_true);
	DVSizeT dvbegin_false(begin_false);
	const DeviceViewable* args[] = { &vec_in, &inds, &vec_true, &vec_false, &dvbegin_in, &dvbegin_true, &dvbegin_false };
	if (!s_for_scatter.launch_n(ctx, n, args)) return (uint32_t)(-1);
	return ret.first;
}

uint32_t TRTC_Partition_Copy_Stencil(TRTCContext& ctx, const DVVectorLike& vec_in, const DVVectorLike& stencil, DVVectorLike& vec_true, DVVectorLike& vec_false, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_true, size_t begin_false)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;

	DVSizeT dvbegin_stencil(begin_stencil);
	Functor src_scan(ctx, { {"stencil", &stencil}, {"begin_stencil", &dvbegin_stencil}, { "pred", &pred } }, { "idx" },
		"        return pred(stencil[idx + begin_stencil]) ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");
	DVVector inds(ctx, "Pair<uint32_t, uint32_t>", n);
	Functor plus(ctx, {}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");
	if (!general_scan(ctx, n, src_scan, inds, plus, 0)) return (uint32_t)(-1);
	Pair<uint32_t, uint32_t> ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + n - 1), sizeof(Pair<uint32_t, uint32_t>));
	
	static TRTC_For s_for_scatter({ "vec_in", "inds", "vec_true", "vec_false", "begin_in", "begin_true", "begin_false" }, "idx",
		"    if ((idx==0 && inds[idx].first>0) || (idx>0 && inds[idx].first>inds[idx-1].first))\n"
		"        vec_true[inds[idx].first -1 + begin_true] = vec_in[idx + begin_in];\n"
		"    else\n"
		"        vec_false[inds[idx].second - 1 + begin_false] = vec_in[idx + begin_in];\n"
	);
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_true(begin_true);
	DVSizeT dvbegin_false(begin_false);
	const DeviceViewable* args[] = { &vec_in, &inds, &vec_true, &vec_false, &dvbegin_in, &dvbegin_true, &dvbegin_false };
	if (!s_for_scatter.launch_n(ctx, n, args)) return (uint32_t)(-1);
	return ret.first;
}

