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

