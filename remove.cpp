#include "copy.h"
#include "remove.h"
#include "general_copy_if.h"

uint32_t TRTC_Remove(DVVectorLike& vec, const DeviceViewable& value)
{
	DVVector cpy(vec.name_elem_cls().c_str(), vec.size());
	TRTC_Copy(vec, cpy);
	Functor src_scan({ {"src", &cpy}, {"value", &value} }, { "idx" },
		"        return src[idx]!=value? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(vec.size(), src_scan, cpy, vec);
}

uint32_t TRTC_Remove_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& value)
{
	Functor src_scan({ {"src", &vec_in}, {"value", &value} }, { "idx" },
		"        return src[idx]!=value? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(vec_in.size(), src_scan, vec_in, vec_out);
}

uint32_t TRTC_Remove_If(DVVectorLike& vec, const Functor& pred)
{
	DVVector cpy(vec.name_elem_cls().c_str(), vec.size());
	TRTC_Copy(vec, cpy);
	Functor src_scan({ {"src", &cpy}, {"pred", &pred} }, { "idx" },
		"        return !pred(src[idx])? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(vec.size(), src_scan, cpy, vec);
}

uint32_t TRTC_Remove_Copy_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred)
{
	Functor src_scan({ {"src", &vec_in}, {"pred", &pred} }, { "idx" },
		"        return !pred(src[idx])? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(vec_in.size(), src_scan, vec_in, vec_out);
}

uint32_t TRTC_Remove_If_Stencil(DVVectorLike& vec, const DVVectorLike& stencil, const Functor& pred)
{
	DVVector cpy(vec.name_elem_cls().c_str(), vec.size());
	TRTC_Copy(vec, cpy);
	Functor src_scan({ {"src", &stencil}, {"pred", &pred} }, { "idx" },
		"        return !pred(src[idx])? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(vec.size(), src_scan, cpy, vec);
}

uint32_t TRTC_Remove_Copy_If_Stencil(const DVVectorLike& vec_in, const DVVectorLike& stencil, DVVectorLike& vec_out, const Functor& pred)
{
	Functor src_scan({ {"src", &stencil}, {"pred", &pred} }, { "idx" },
		"        return !pred(src[idx])? (uint32_t)1:(uint32_t)0;\n");
	return general_copy_if(vec_in.size(), src_scan, vec_in, vec_out);
}