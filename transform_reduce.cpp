#include <memory.h>
#include "transform_reduce.h"
#include "general_reduce.h"

bool TRTC_Transform_Reduce(TRTCContext& ctx, const DVVectorLike& vec, const Functor& unary_op, const DeviceViewable& init, const Functor& binary_op, ViewBuf& ret, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);

	Functor src(ctx, { {"vec", &vec}, {"begin", &dvbegin }, { "init", &init }, {"unary_op", &unary_op} }, { "idx" },
		"        return idx>0?unary_op(vec[idx - 1 + begin]):init;\n");
	if (end == (size_t)(-1)) end = vec.size();
	end++;
	size_t ret_size = ctx.size_of(init.name_view_cls().c_str());
	ret.resize(ret_size);
	memset(ret.data(), 0, ret_size);
	if (end - begin < 1) return true;
	if (!general_reduce(ctx, end - begin, init.name_view_cls().c_str(), src, binary_op, ret)) return false;
	return true;
}
