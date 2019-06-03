#include <memory.h>
#include "reduce.h"
#include "general_reduce.h"

bool TRTC_Reduce(TRTCContext& ctx, const DVVectorLike& vec, ViewBuf& ret, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);

	Functor src(ctx, { {"vec_in", &vec}, {"begin", &dvbegin } }, { "idx" },
		"        return vec_in[idx + begin];\n");

	Functor op("Plus");

	if (end == (size_t)(-1)) end = vec.size();

	ret.resize(vec.elem_size());
	memset(ret.data(), 0, vec.elem_size());
	if (end - begin < 1) return true;
	if (!general_reduce(ctx, end - begin, vec.name_elem_cls().c_str(), src, op, ret)) return false;
	return true;
}

bool TRTC_Reduce(TRTCContext& ctx, const DVVectorLike& vec, const DeviceViewable& init, ViewBuf& ret, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);
	
	Functor src(ctx, { {"vec_in", &vec}, {"begin", &dvbegin }, {"init", &init} }, { "idx" },
		"        return idx>0 ? vec_in[idx - 1 + begin] : (decltype(vec_in)::value_t)init;\n");

	Functor op("Plus");

	if (end == (size_t)(-1)) end = vec.size();
	end++;
	ret.resize(vec.elem_size());
	memset(ret.data(), 0, vec.elem_size());
	if (end - begin < 1) return true;
	if (!general_reduce(ctx, end - begin, vec.name_elem_cls().c_str(), src, op, ret)) return false;
	return true;
}

bool TRTC_Reduce(TRTCContext& ctx, const DVVectorLike& vec, const DeviceViewable& init, const Functor& binary_op, ViewBuf& ret, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);
	Functor src(ctx, { {"vec_in", &vec}, {"begin", &dvbegin }, {"init", &init} }, { "idx" },
		"        return idx>0 ? vec_in[idx - 1 + begin] : (decltype(vec_in)::value_t)init;\n");
	if (end == (size_t)(-1)) end = vec.size();
	end++;
	ret.resize(vec.elem_size());
	memset(ret.data(), 0, vec.elem_size());
	if (end - begin < 1) return true;
	if (!general_reduce(ctx, end - begin, vec.name_elem_cls().c_str(), src, binary_op, ret)) return false;
	return true;
}

#include "scan.h"
#include "general_copy_if.h"

uint32_t TRTC_Reduce_By_Key(TRTCContext& ctx, const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out, size_t begin_key_in, size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out)
{
	if (end_key_in == (size_t)(-1)) end_key_in = key_in.size();
	size_t n = end_key_in - begin_key_in;
	DVVector scan_dst(ctx, value_out.name_elem_cls().c_str(), n);
	TRTC_Inclusive_Scan_By_Key(ctx, key_in, value_in, scan_dst, begin_key_in, end_key_in, begin_value_in, 0);
	
	DVSizeT dvbegin_key_in(begin_key_in);
	DVSizeT dv_n(n);
	Functor src_scan(ctx, { {"key_in", &key_in}, {"begin_key_in", &dvbegin_key_in}, {"n", &dv_n } }, { "idx" },
		"        return  idx==n-1 || key_in[idx+begin_key_in]!=key_in[idx+begin_key_in+1] ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, key_in, scan_dst, key_out, value_out, begin_key_in, 0, begin_key_out, begin_value_out);

}

uint32_t TRTC_Reduce_By_Key(TRTCContext& ctx, const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out, const Functor& binary_pred, size_t begin_key_in , size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out)
{
	if (end_key_in == (size_t)(-1)) end_key_in = key_in.size();
	size_t n = end_key_in - begin_key_in;
	DVVector scan_dst(ctx, value_out.name_elem_cls().c_str(), n);
	TRTC_Inclusive_Scan_By_Key(ctx, key_in, value_in, scan_dst, binary_pred, begin_key_in, end_key_in, begin_value_in, 0);

	DVSizeT dvbegin_key_in(begin_key_in);
	DVSizeT dv_n(n);
	Functor src_scan(ctx, { {"key_in", &key_in}, {"begin_key_in", &dvbegin_key_in}, {"n", &dv_n },  {"binary_pred", &binary_pred} }, { "idx" },
		"        return  idx==n-1 || !binary_pred(key_in[idx+begin_key_in],key_in[idx+begin_key_in+1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, key_in, scan_dst, key_out, value_out, begin_key_in, 0, begin_key_out, begin_value_out);
}

uint32_t TRTC_Reduce_By_Key(TRTCContext& ctx, const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out, const Functor& binary_pred, const Functor& binary_op, size_t begin_key_in, size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out)
{
	if (end_key_in == (size_t)(-1)) end_key_in = key_in.size();
	size_t n = end_key_in - begin_key_in;
	DVVector scan_dst(ctx, value_out.name_elem_cls().c_str(), n);
	TRTC_Inclusive_Scan_By_Key(ctx, key_in, value_in, scan_dst, binary_pred, binary_op, begin_key_in, end_key_in, begin_value_in, 0);

	DVSizeT dvbegin_key_in(begin_key_in);
	DVSizeT dv_n(n);
	Functor src_scan(ctx, { {"key_in", &key_in}, {"begin_key_in", &dvbegin_key_in}, {"n", &dv_n },  {"binary_pred", &binary_pred} }, { "idx" },
		"        return  idx==n-1 || !binary_pred(key_in[idx+begin_key_in], key_in[idx+begin_key_in+1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, key_in, scan_dst, key_out, value_out, begin_key_in, 0, begin_key_out, begin_value_out);
}
