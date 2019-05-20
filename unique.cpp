#include "copy.h"
#include "unique.h"
#include "general_copy_if.h"

uint32_t TRTC_Unique(TRTCContext& ctx, DVVectorLike& vec, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	size_t n = end - begin;
	DVVector cpy(ctx, vec.name_elem_cls().c_str(), n);
	TRTC_Copy(ctx, vec, cpy, begin, end);
	Functor src_scan(ctx, { {"src", &cpy} }, { "idx" },
		"        return  idx==0 || src[idx]!=src[idx-1] ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, cpy, vec, 0, begin);
}

uint32_t TRTC_Unique(TRTCContext& ctx, DVVectorLike& vec, const Functor& binary_pred, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	size_t n = end - begin;
	DVVector cpy(ctx, vec.name_elem_cls().c_str(), n);
	TRTC_Copy(ctx, vec, cpy, begin, end);
	Functor src_scan(ctx, { {"src", &cpy}, {"binary_pred", &binary_pred} }, { "idx" },
		"        return  idx==0 || !binary_pred(src[idx],src[idx-1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, cpy, vec, 0, begin);
}

uint32_t TRTC_Unique_Copy(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src_scan(ctx, { {"src", &vec_in}, {"begin_in", &dvbegin_in} }, { "idx" },
		"        return  idx==0 || src[idx+begin_in]!=src[idx+begin_in-1] ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, vec_in, vec_out, begin_in, begin_out);
}

uint32_t TRTC_Unique_Copy(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_pred, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	size_t n = end_in - begin_in;
	DVSizeT dvbegin_in(begin_in);
	Functor src_scan(ctx, { {"src", &vec_in}, {"begin_in", &dvbegin_in},  {"binary_pred", &binary_pred} }, { "idx" },
		"        return  idx==0 || !binary_pred(src[idx+begin_in],src[idx+begin_in-1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, vec_in, vec_out, begin_in, begin_out);
}

uint32_t TRTC_Unique_By_Key(TRTCContext& ctx, DVVectorLike& keys, DVVectorLike& values, size_t begin_key , size_t end_key, size_t begin_value)
{
	if (end_key == (size_t)(-1)) end_key = keys.size();
	size_t n = end_key - begin_key;
	DVVector cpy_keys(ctx, keys.name_elem_cls().c_str(), n);
	DVVector cpy_values(ctx, values.name_elem_cls().c_str(), n);
	TRTC_Copy(ctx, keys, cpy_keys, begin_key, end_key);
	TRTC_Copy(ctx, values, cpy_values, begin_value, begin_value+n);
	Functor src_scan(ctx, { {"src", &cpy_keys} }, { "idx" },
		"        return  idx==0 || src[idx]!=src[idx-1] ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, cpy_keys, cpy_values, keys, values, 0, 0, begin_key, begin_value);
}

uint32_t TRTC_Unique_By_Key(TRTCContext& ctx, DVVectorLike& keys, DVVectorLike& values, const Functor& binary_pred, size_t begin_key, size_t end_key, size_t begin_value)
{
	if (end_key == (size_t)(-1)) end_key = keys.size();
	size_t n = end_key - begin_key;
	DVVector cpy_keys(ctx, keys.name_elem_cls().c_str(), n);
	DVVector cpy_values(ctx, values.name_elem_cls().c_str(), n);
	TRTC_Copy(ctx, keys, cpy_keys, begin_key, end_key);
	TRTC_Copy(ctx, values, cpy_values, begin_value, begin_value + n);
	Functor src_scan(ctx, { {"src", &cpy_keys}, {"binary_pred", &binary_pred} }, { "idx" },
		"        return  idx==0 || !binary_pred(src[idx],src[idx-1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, cpy_keys, cpy_values, keys, values, 0, 0, begin_key, begin_value);
}

uint32_t TRTC_Unique_By_Key_Copy(TRTCContext& ctx, const DVVectorLike& keys_in, const DVVectorLike& values_in, DVVectorLike& keys_out, DVVectorLike& values_out, size_t begin_key_in, size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out)
{
	if (end_key_in == (size_t)(-1)) end_key_in = keys_in.size();
	size_t n = end_key_in - begin_key_in;
	DVSizeT dvbegin_key_in(begin_key_in);
	Functor src_scan(ctx, { {"src", &keys_in}, {"begin_in", &dvbegin_key_in} }, { "idx" },
		"        return  idx==0 || src[idx+begin_in]!=src[idx+begin_in-1] ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, keys_in, values_in, keys_out, values_out, begin_key_in, begin_value_in, begin_key_out, begin_value_out);
}


uint32_t TRTC_Unique_By_Key_Copy(TRTCContext& ctx, const DVVectorLike& keys_in, const DVVectorLike& values_in, DVVectorLike& keys_out, DVVectorLike& values_out, const Functor& binary_pred, size_t begin_key_in, size_t end_key_in, size_t begin_value_in, size_t begin_key_out, size_t begin_value_out)
{
	if (end_key_in == (size_t)(-1)) end_key_in = keys_in.size();
	size_t n = end_key_in - begin_key_in;
	DVSizeT dvbegin_key_in(begin_key_in);
	Functor src_scan(ctx, { {"src", &keys_in}, {"begin_in", &dvbegin_key_in},  {"binary_pred", &binary_pred} }, { "idx" },
		"        return  idx==0 || !binary_pred(src[idx+begin_in],src[idx+begin_in-1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(ctx, n, src_scan, keys_in, values_in, keys_out, values_out, begin_key_in, begin_value_in, begin_key_out, begin_value_out);
}
