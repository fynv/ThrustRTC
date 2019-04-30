#include "reduce.h"
#include "general_reduce.h"

bool TRTC_Reduce(TRTCContext& ctx, const DVVectorLike& vec, ViewBuf& ret, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);

	Functor src = { { {"vec_in", &vec}, {"begin", &dvbegin} } , { "idx" }, "ret",
		"        ret = vec_in[idx + begin];\n" };

	Functor op = { {},{ "x", "y" }, "ret", "        ret = x + y;\n" };

	if (end == (size_t)(-1)) end = vec.size();

	ret.resize(vec.elem_size());
	memset(ret.data(), 0, vec.elem_size());
	if (end - begin < 1) return true;
	if (!general_reduce(ctx, end - begin, vec.name_elem_cls().c_str(), src, op, ret)) return false;
	return true;
}

bool TRTC_Reduce(TRTCContext& ctx, const DVVectorLike& vec, const DeviceViewable& init, ViewBuf& ret, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);
	Functor src = { { {"vec_in", &vec}, {"begin", &dvbegin}, {"init", &init} } , { "idx" }, "ret",
		"        ret = idx>0 ? vec_in[idx - 1 + begin] : (decltype(vec_in)::value_t)init;\n" };
	Functor op = { {},{ "x", "y" }, "ret", "        ret = x + y;\n" };
	if (end == (size_t)(-1)) end = vec.size();
	end++;
	ret.resize(vec.elem_size());
	memset(ret.data(), 0, vec.elem_size());
	if (end - begin < 1) return true;
	if (!general_reduce(ctx, end - begin, vec.name_elem_cls().c_str(), src, op, ret)) return false;
	return true;
}

bool TRTC_Reduce(TRTCContext& ctx, const DVVectorLike& vec, const DeviceViewable& init, const Functor& binary_op, ViewBuf& ret, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);
	Functor src = { { {"_vec_in", &vec}, {"_begin", &dvbegin}, {"_init", &init} } , { "_idx" }, "_ret",
		"        _ret = _idx>0 ? _vec_in[_idx - 1 + _begin] : (decltype(_vec_in)::value_t)_init;\n" };
	if (end == (size_t)(-1)) end = vec.size();
	end++;
	ret.resize(vec.elem_size());
	memset(ret.data(), 0, vec.elem_size());
	if (end - begin < 1) return true;
	if (!general_reduce(ctx, end - begin, vec.name_elem_cls().c_str(), src, binary_op, ret)) return false;
	return true;
}

