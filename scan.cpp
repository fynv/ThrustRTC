#include "scan.h"
#include "general_scan.h"

bool TRTC_Inclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src(ctx, { {"vec_in", &vec_in}, {"begin_in", &dvbegin_in } }, { "idx" },
		"        return vec_in[idx + begin_in];\n");
	Functor plus("Plus");

	return general_scan(ctx, n, src, vec_out, plus, begin_out);
}

bool TRTC_Inclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_op, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src(ctx, { {"vec_in", &vec_in}, {"begin_in", &dvbegin_in } }, { "idx" },
		"        return vec_in[idx + begin_in];\n");
	return general_scan(ctx, n, src, vec_out, binary_op, begin_out);
}


bool TRTC_Exclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src(ctx, { {"vec_in", &vec_in}, {"begin_in", &dvbegin_in } }, { "idx" },
		"        return idx>0? vec_in[idx - 1 + begin_in] : (decltype(vec_in)::value_t) 0;\n");
	Functor plus("Plus");
	return general_scan(ctx, n, src, vec_out, plus, begin_out);
}

bool TRTC_Exclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& init, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src(ctx, { {"vec_in", &vec_in}, {"begin_in", &dvbegin_in }, {"init", &init} }, { "idx" },
		"        return idx>0? vec_in[idx - 1 + begin_in] : (decltype(vec_in)::value_t)init;\n");
	Functor plus("Plus");
	return general_scan(ctx, n, src, vec_out, plus, begin_out);
}

bool TRTC_Exclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& init, const Functor& binary_op, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src(ctx, { {"vec_in", &vec_in},{"begin_in", &dvbegin_in }, {"init", &init} }, { "idx" },
		"        return idx>0? vec_in[idx - 1 + begin_in] : (decltype(vec_in)::value_t)init;\n");
	return general_scan(ctx, n, src, vec_out, binary_op, begin_out);
}

