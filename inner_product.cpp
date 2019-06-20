#include <memory.h>
#include "inner_product.h"
#include "general_reduce.h"

bool TRTC_Inner_Product(const DVVectorLike& vec1, const DVVectorLike& vec2, const DeviceViewable& init, ViewBuf& ret)
{
	Functor src({ {"vec1", &vec1}, {"vec2", &vec2}, {"init", &init} }, { "idx" },
		"        return idx>0 ? vec1[idx - 1] * vec2[idx - 1] : init;\n");
	Functor op("Plus");

	size_t ret_size = TRTC_Size_Of(init.name_view_cls().c_str());
	ret.resize(ret_size);
	memset(ret.data(), 0, ret_size);
	if (!general_reduce(vec1.size()+1, init.name_view_cls().c_str(), src, op, ret)) return false;
	return true;
}

bool TRTC_Inner_Product(const DVVectorLike& vec1, const DVVectorLike& vec2, const DeviceViewable& init, ViewBuf& ret, const Functor& binary_op1, const Functor& binary_op2)
{
	Functor src({ {"vec1", &vec1}, {"vec2", &vec2}, {"init", &init}, {"binary_op2", &binary_op2} }, { "idx" },
		"        return idx>0 ? binary_op2(vec1[idx - 1], vec2[idx - 1]) : init;\n");

	size_t ret_size = TRTC_Size_Of(init.name_view_cls().c_str());
	ret.resize(ret_size);
	memset(ret.data(), 0, ret_size);
	if (!general_reduce(vec1.size()+1, init.name_view_cls().c_str(), src, binary_op1, ret)) return false;
	return true;
	
}


