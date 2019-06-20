#include <memory.h>
#include "transform_reduce.h"
#include "general_reduce.h"

bool TRTC_Transform_Reduce(const DVVectorLike& vec, const Functor& unary_op, const DeviceViewable& init, const Functor& binary_op, ViewBuf& ret)
{
	Functor src({ {"vec", &vec}, { "init", &init }, {"unary_op", &unary_op} }, { "idx" },
		"        return idx>0?unary_op(vec[idx - 1]):init;\n");
	size_t ret_size = TRTC_Size_Of(init.name_view_cls().c_str());
	ret.resize(ret_size);
	memset(ret.data(), 0, ret_size);
	if (!general_reduce(vec.size()+1, init.name_view_cls().c_str(), src, binary_op, ret)) return false;
	return true;
}
