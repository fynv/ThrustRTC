#include "extrema.h"
#include "general_reduce.h"

bool TRTC_Min_Element(TRTCContext& ctx, const DVVectorLike& vec, size_t& id_min, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);
	Functor src = { { {"begin", &dvbegin} } , { "idx" }, "ret",
		"        ret = begin + idx;\n" };
	Functor op = { { {"vec", &vec} },{ "i1", "i2" }, "ret", "        ret = vec[i2]<vec[i1]?i2:i1;\n" };

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(ctx, end - begin, "size_t", src, op, buf)) return false;
	id_min = *(size_t*)buf.data();
	return true;
}

bool TRTC_Min_Element(TRTCContext& ctx, const DVVectorLike& vec, const Functor& comp, size_t& id_min, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);

	Functor src = { { {"begin", &dvbegin} } , { "idx" }, "ret",
		"        ret = begin + idx;\n" };

	std::vector<TRTCContext::AssignedParam> arg_map = comp.arg_map;
	arg_map.push_back({ "_vec", &vec });

	std::string body_func_str = comp.generate_code("bool", { "_vec[_i2]", "_vec[_i1]" }) +
		"        _ret = " + comp.functor_ret + "?_i2:_i1;\n";

	Functor op = { arg_map, { "_i1", "_i2" }, "_ret", body_func_str.c_str() };

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(ctx, end - begin, "size_t", src, op, buf)) return false;
	id_min = *(size_t*)buf.data();
	return true;
}


bool TRTC_Max_Element(TRTCContext& ctx, const DVVectorLike& vec, size_t& id_max, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);
	Functor src = { { {"begin", &dvbegin} } , { "idx" }, "ret",
		"        ret = begin + idx;\n" };
	Functor op = { { {"vec", &vec} },{ "i1", "i2" }, "ret", "        ret = vec[i1]<vec[i2]?i2:i1;\n" };

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(ctx, end - begin, "size_t", src, op, buf)) return false;
	id_max = *(size_t*)buf.data();
	return true;
}


bool TRTC_Max_Element(TRTCContext& ctx, const DVVectorLike& vec, const Functor& comp, size_t& id_max, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);

	Functor src = { { {"begin", &dvbegin} } , { "idx" }, "ret",
		"        ret = begin + idx;\n" };

	std::vector<TRTCContext::AssignedParam> arg_map = comp.arg_map;
	arg_map.push_back({ "_vec", &vec });

	std::string body_func_str = comp.generate_code("bool", { "_vec[_i1]", "_vec[_i2]" }) +
		"        _ret = " + comp.functor_ret + "?_i2:_i1;\n";

	Functor op = { arg_map, { "_i1", "_i2" }, "_ret", body_func_str.c_str() };

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(ctx, end - begin, "size_t", src, op, buf)) return false;
	id_max = *(size_t*)buf.data();
	return true;
}

bool TRTC_MinMax_Element(TRTCContext& ctx, const DVVectorLike& vec, size_t& id_min, size_t& id_max, size_t begin, size_t end)
{
	struct MinMaxIds
	{
		size_t id_min;
		size_t id_max;
	};

	std::string d_MinMaxIds = ctx.add_custom_struct(
		"    size_t id_min;\n"
		"    size_t id_max;\n"
	);

	DVSizeT dvbegin(begin);
	Functor src = { { {"begin", &dvbegin} } , { "idx" }, "ret",
		"        ret = {begin + idx, begin + idx};\n" };
	Functor op = { { {"vec", &vec} },{ "i1", "i2" }, "ret", "        ret = { vec[i2.id_min]<vec[i1.id_min]?i2.id_min:i1.id_min, vec[i1.id_max]<vec[i2.id_max]?i2.id_max:i1.id_max};\n" };

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(ctx, end - begin, d_MinMaxIds.c_str(), src, op, buf)) return false;
	MinMaxIds res = *(MinMaxIds*)buf.data();
	id_min = res.id_min;
	id_max = res.id_max;
	return true;
}

bool TRTC_MinMax_Element(TRTCContext& ctx, const DVVectorLike& vec, const Functor& comp, size_t& id_min, size_t& id_max, size_t begin, size_t end)
{
	struct MinMaxIds
	{
		size_t id_min;
		size_t id_max;
	};

	std::string d_MinMaxIds = ctx.add_custom_struct(
		"    size_t id_min;\n"
		"    size_t id_max;\n"
	);

	DVSizeT dvbegin(begin);
	Functor src = { { {"begin", &dvbegin} } , { "idx" }, "ret",
		"        ret = {begin + idx, begin + idx};\n" };

	std::vector<TRTCContext::AssignedParam> arg_map = comp.arg_map;
	arg_map.push_back({ "_vec", &vec });

	std::string body_func_str = std::string("    bool _ret1, _ret2;\n") +
		"    {\n" + comp.generate_code("bool", { "_vec[_i2.id_min]", "_vec[_i1.id_min]" }) +
		"        _ret1 = " + comp.functor_ret + ";\n    }\n"
		"    {\n" + comp.generate_code("bool", { "_vec[_i1.id_max]", "_vec[_i2.id_max]" }) +
		"        _ret2 = " + comp.functor_ret + ";\n    }\n"
		"        _ret = { _ret1 ?_i2.id_min:_i1.id_min, _ret2 ?_i2.id_max:_i1.id_max};\n";

	Functor op = { arg_map, { "_i1", "_i2" }, "_ret", body_func_str.c_str() };

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(ctx, end - begin, d_MinMaxIds.c_str(), src, op, buf)) return false;
	MinMaxIds res = *(MinMaxIds*)buf.data();
	id_min = res.id_min;
	id_max = res.id_max;
	return true;
}

