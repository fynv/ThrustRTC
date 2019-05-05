#include "scan.h"
#include "general_scan.h"

bool TRTC_Inclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src = { { {"_vec_in", &vec_in}, {"_begin_in", &dvbegin_in} } , { "_idx" }, "_ret",
		"        _ret = (decltype(_ret)) _vec_in[_idx + _begin_in];\n" };
	Functor plus = { {},{ "x", "y" }, "ret", "        ret = x + y;\n" };
	return general_scan(ctx, n, src, vec_out, plus, begin_out);
}

bool TRTC_Inclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_op, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src = { { {"_vec_in", &vec_in}, {"_begin_in", &dvbegin_in} } , { "_idx" }, "_ret",
		"        _ret = (decltype(_ret)) _vec_in[_idx + _begin_in];\n" };
	return general_scan(ctx, n, src, vec_out, binary_op, begin_out);
}


bool TRTC_Exclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src = { { {"_vec_in", &vec_in}, {"_begin_in", &dvbegin_in}} , { "_idx" }, "_ret",
		"        _ret = _idx>0?  (decltype(_ret))_vec_in[_idx - 1 + _begin_in] :  (decltype(_ret))0;\n" };
	Functor plus = { {},{ "x", "y" }, "ret", "        ret = x + y;\n" };
	return general_scan(ctx, n, src, vec_out, plus, begin_out);
}

bool TRTC_Exclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& init, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src = { { {"_vec_in", &vec_in}, {"_begin_in", &dvbegin_in}, {"_init", &init} } , { "_idx" }, "_ret",
		"        _ret = _idx>0?  (decltype(_ret))_vec_in[_idx - 1 + _begin_in] :  (decltype(_ret))_init;\n" };
	Functor plus = { {},{ "x", "y" }, "ret", "        ret = x + y;\n" };
	return general_scan(ctx, n, src, vec_out, plus, begin_out);
}

bool TRTC_Exclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& init, const Functor& binary_op, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src = { { {"_vec_in", &vec_in}, {"_begin_in", &dvbegin_in}, {"_init", &init} } , { "_idx" }, "_ret",
		"        _ret = _idx>0?  (decltype(_ret))_vec_in[_idx - 1 + _begin_in] :  (decltype(_ret))_init;\n" };
	return general_scan(ctx, n, src, vec_out, binary_op, begin_out);
}

