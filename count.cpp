#include "count.h"
#include "general_reduce.h"

bool TRTC_Count(TRTCContext& ctx, const DVVectorLike& vec, const DeviceViewable& value, size_t& ret, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);

	Functor src = { { {"vec_in", &vec}, {"eq_value", &value }, {"begin", &dvbegin} } , { "idx" }, "ret", 
		"        ret = (vec_in[idx + begin] == (decltype(vec_in)::value_t)eq_value)?1:0;\n" };

	Functor op  = { {},{ "x", "y" }, "ret", "        ret = x + y;\n" };

	if (end == (size_t)(-1)) end = vec.size();

	ret = 0;
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(ctx, end - begin, "size_t", src, op, buf)) return false;
	ret = *(size_t*)buf.data();
	return true;
}

bool TRTC_Count_If(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, size_t& ret, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);

	std::vector<TRTCContext::AssignedParam> arg_map = pred.arg_map;
	arg_map.push_back({ "_vec_in", &vec });
	arg_map.push_back({ "_begin", &dvbegin });

	std::string body_func_str = pred.generate_code("bool", { "_vec_in[_idx + _begin]" }) +
		"        _ret = " + pred.functor_ret + "? 1:0;\n";

	Functor src = { arg_map, { "_idx" }, "_ret", body_func_str.c_str()};
	Functor op = { {},{ "x", "y" }, "ret", "        ret = x + y;\n" };
	if (end == (size_t)(-1)) end = vec.size();

	ret = 0;
	if (end - begin < 1) return true;
	ViewBuf buf;
	if (!general_reduce(ctx, end - begin, "size_t", src, op, buf)) return false;
	ret = *(size_t*)buf.data();
	return true;

}