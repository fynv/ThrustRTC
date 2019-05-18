#include "copy.h"
#include "remove.h"
#include "general_copy_if.h"

uint32_t TRTC_Remove(TRTCContext& ctx, DVVectorLike& vec, const DeviceViewable& value, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	size_t n = end - begin;
	DVVector cpy(ctx, vec.name_elem_cls().c_str(), n);
	TRTC_Copy(ctx, vec, cpy, begin, end);
	Functor src_scan(ctx, { {"src", &cpy}, {"value", &value} }, { "idx" },
		"        return src[idx]!=value? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, cpy, vec, 0, begin);
}

uint32_t TRTC_Remove_Copy(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& value, size_t begin_in , size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src_scan(ctx, { {"src", &vec_in}, {"begin_src", &dvbegin_in}, {"value", &value} }, { "idx" },
		"        return src[idx + begin_src]!=value? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, vec_in, vec_out, begin_in, begin_out);
}

uint32_t TRTC_Remove_If(TRTCContext& ctx, DVVectorLike& vec, const Functor& pred, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	size_t n = end - begin;
	DVVector cpy(ctx, vec.name_elem_cls().c_str(), n);
	TRTC_Copy(ctx, vec, cpy, begin, end);
	Functor src_scan(ctx, { {"src", &cpy}, {"pred", &pred} }, { "idx" },
		"        return !pred(src[idx])? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, cpy, vec, 0, begin);
}

uint32_t TRTC_Remove_Copy_If(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src_scan(ctx, { {"src", &vec_in}, {"begin_src", &dvbegin_in}, {"pred", &pred} }, { "idx" },
		"        return !pred(src[idx + begin_src])? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, vec_in, vec_out, begin_in, begin_out);
}

uint32_t TRTC_Remove_If_Stencil(TRTCContext& ctx, DVVectorLike& vec, const DVVectorLike& stencil, const Functor& pred, size_t begin, size_t end, size_t begin_stencil)
{
	if (end == (size_t)(-1)) end = vec.size();
	size_t n = end - begin;
	DVVector cpy(ctx, vec.name_elem_cls().c_str(), n);
	TRTC_Copy(ctx, vec, cpy, begin, end);
	DVSizeT dvbegin_stencil(begin_stencil);
	Functor src_scan(ctx, { {"src", &stencil},  {"begin_src", &dvbegin_stencil}, {"pred", &pred} }, { "idx" },
		"        return !pred(src[idx + begin_src])? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, cpy, vec, 0, begin);
}

uint32_t TRTC_Remove_Copy_If_Stencil(TRTCContext& ctx, const DVVectorLike& vec_in, const DVVectorLike& stencil, DVVectorLike& vec_out, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_stencil(begin_stencil);
	Functor src_scan(ctx, { {"src", &stencil}, {"begin_src", &dvbegin_stencil}, {"pred", &pred} }, { "idx" },
		"        return !pred(src[idx + begin_src])? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, vec_in, vec_out, begin_in, begin_out);
}