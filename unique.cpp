#include "copy.h"
#include "unique.h"
#include "general_copy_if.h"

uint32_t TRTC_Unique(DVVectorLike& vec)
{
	DVVector cpy(vec.name_elem_cls().c_str(), vec.size());
	TRTC_Copy(vec, cpy);
	Functor src_scan({ {"src", &cpy} }, { "idx" },
		"        return  idx==0 || src[idx]!=src[idx-1] ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(vec.size(), src_scan, cpy, vec);
}

uint32_t TRTC_Unique(DVVectorLike& vec, const Functor& binary_pred)
{
	DVVector cpy(vec.name_elem_cls().c_str(), vec.size());
	TRTC_Copy(vec, cpy);
	Functor src_scan({ {"src", &cpy}, {"binary_pred", &binary_pred} }, { "idx" },
		"        return  idx==0 || !binary_pred(src[idx],src[idx-1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(vec.size(), src_scan, cpy, vec);
}

uint32_t TRTC_Unique_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out)
{
	Functor src_scan({ {"src", &vec_in} }, { "idx" },
		"        return  idx==0 || src[idx]!=src[idx-1] ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(vec_in.size(), src_scan, vec_in, vec_out);
}

uint32_t TRTC_Unique_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_pred)
{
	Functor src_scan({ {"src", &vec_in},  {"binary_pred", &binary_pred} }, { "idx" },
		"        return  idx==0 || !binary_pred(src[idx],src[idx-1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(vec_in.size(), src_scan, vec_in, vec_out);
}

uint32_t TRTC_Unique_By_Key(DVVectorLike& keys, DVVectorLike& values)
{
	DVVector cpy_keys(keys.name_elem_cls().c_str(), keys.size());
	DVVector cpy_values(values.name_elem_cls().c_str(), values.size());
	TRTC_Copy(keys, cpy_keys);
	TRTC_Copy(values, cpy_values);
	Functor src_scan({ {"src", &cpy_keys} }, { "idx" },
		"        return  idx==0 || src[idx]!=src[idx-1] ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(keys.size(), src_scan, cpy_keys, cpy_values, keys, values);
}

uint32_t TRTC_Unique_By_Key(DVVectorLike& keys, DVVectorLike& values, const Functor& binary_pred)
{
	DVVector cpy_keys(keys.name_elem_cls().c_str(), keys.size());
	DVVector cpy_values(values.name_elem_cls().c_str(), values.size());
	TRTC_Copy(keys, cpy_keys);
	TRTC_Copy(values, cpy_values);
	Functor src_scan({ {"src", &cpy_keys}, {"binary_pred", &binary_pred} }, { "idx" },
		"        return  idx==0 || !binary_pred(src[idx],src[idx-1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(keys.size(), src_scan, cpy_keys, cpy_values, keys, values);
}

uint32_t TRTC_Unique_By_Key_Copy(const DVVectorLike& keys_in, const DVVectorLike& values_in, DVVectorLike& keys_out, DVVectorLike& values_out)
{
	Functor src_scan({ {"src", &keys_in} }, { "idx" },
		"        return  idx==0 || src[idx]!=src[idx-1] ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(keys_in.size(), src_scan, keys_in, values_in, keys_out, values_out);
}


uint32_t TRTC_Unique_By_Key_Copy(const DVVectorLike& keys_in, const DVVectorLike& values_in, DVVectorLike& keys_out, DVVectorLike& values_out, const Functor& binary_pred)
{
	Functor src_scan({ {"src", &keys_in},  {"binary_pred", &binary_pred} }, { "idx" },
		"        return  idx==0 || !binary_pred(src[idx],src[idx-1]) ? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(keys_in.size(), src_scan, keys_in, values_in, keys_out, values_out);
}
