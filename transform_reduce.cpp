#include <memory.h>
#include "transform_reduce.h"
#include "general_reduce.h"

bool TRTC_Transform_Reduce(TRTCContext& ctx, const DVVectorLike& vec, const Functor& unary_op, const DeviceViewable& init, const Functor& binary_op, ViewBuf& ret, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);
	std::vector<TRTCContext::AssignedParam> arg_map_src = unary_op.arg_map;
	arg_map_src.push_back({ "_vec", &vec });
	arg_map_src.push_back({ "_begin", &dvbegin });
	arg_map_src.push_back({ "_init", &init });

	std::string body_func_src =
		std::string("    if(_idx<1)\n    {\n") +
		"        _ret = _init;\n    }\n" +
		"    else\n    {\n" +
		unary_op.generate_code(init.name_view_cls().c_str(), { "_vec[_idx - 1 + _begin]" }) +
		"        _ret = " + unary_op.functor_ret + ";\n    }\n";

	Functor src = { arg_map_src, { "_idx" }, "_ret", body_func_src.c_str() };
	if (end == (size_t)(-1)) end = vec.size();
	end++;
	size_t ret_size = ctx.size_of(init.name_view_cls().c_str());
	ret.resize(ret_size);
	memset(ret.data(), 0, ret_size);
	if (end - begin < 1) return true;
	if (!general_reduce(ctx, end - begin, init.name_view_cls().c_str(), src, binary_op, ret)) return false;
	return true;
}
