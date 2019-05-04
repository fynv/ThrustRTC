#include "inner_product.h"
#include "general_reduce.h"

bool TRTC_Inner_Product(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, const DeviceViewable& init, ViewBuf& ret, size_t begin1, size_t end1, size_t begin2)
{
	DVSizeT dvbegin1(begin1);
	DVSizeT dvbegin2(begin2);
	Functor src = { { {"vec1", &vec1}, {"vec2", &vec2}, {"begin1", &dvbegin1}, {"begin2", &dvbegin2}, {"init", &init} } , { "idx" }, "ret",
		"        ret = idx>0 ? vec1[idx - 1 + begin1] * vec2[idx - 1 + begin2] : init;\n" };
	Functor op = { {},{ "x", "y" }, "ret", "        ret = x + y;\n" };
	if (end1 == (size_t)(-1)) end1 = vec1.size();
	end1++;
	size_t ret_size = ctx.size_of(init.name_view_cls().c_str());
	ret.resize(ret_size);
	memset(ret.data(), 0, ret_size);
	if (end1 - begin1 < 1) return true;
	if (!general_reduce(ctx, end1 - begin1, init.name_view_cls().c_str(), src, op, ret)) return false;
	return true;
}

bool TRTC_Inner_Product(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, const DeviceViewable& init, ViewBuf& ret, const Functor& binary_op1, const Functor& binary_op2, size_t begin1, size_t end1, size_t begin2)
{
	DVSizeT dvbegin1(begin1);
	DVSizeT dvbegin2(begin2);

	std::vector<TRTCContext::AssignedParam> arg_map_src = binary_op2.arg_map;
	arg_map_src.push_back({ "_vec1", &vec1 });
	arg_map_src.push_back({ "_vec2", &vec2 });
	arg_map_src.push_back({ "_begin1", &dvbegin1 });
	arg_map_src.push_back({ "_begin2", &dvbegin2 });
	arg_map_src.push_back({ "_init", &init });

	std::string body_func_src =
		std::string("    if(_idx<1)\n    {\n") +
		"        _ret = _init;\n    }\n" +
		"    else\n    {\n" +
		binary_op2.generate_code(init.name_view_cls().c_str(), { "_vec1[_idx - 1 + _begin1]", "_vec2[_idx - 1 + _begin2]" }) +
		"        _ret = " + binary_op2.functor_ret + ";\n    }\n";
	   
	Functor src = { arg_map_src, { "_idx" }, "_ret", body_func_src.c_str() };
	if (end1 == (size_t)(-1)) end1 = vec1.size();
	end1++;
	size_t ret_size = ctx.size_of(init.name_view_cls().c_str());
	ret.resize(ret_size);
	memset(ret.data(), 0, ret_size);
	if (end1 - begin1 < 1) return true;
	if (!general_reduce(ctx, end1 - begin1, init.name_view_cls().c_str(), src, binary_op1, ret)) return false;
	return true;
	
}


