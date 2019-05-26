#include "find.h"
#include "general_find.h"

bool TRTC_Find(TRTCContext& ctx, const DVVectorLike& vec, const DeviceViewable& value, size_t& result, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	Functor src(ctx, { {"vec", &vec}, {"value", &value} }, { "id" }, "        return vec[id]==value;\n");
	return general_find(ctx, begin, end, src, result);
}

bool TRTC_Find_If(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, size_t& result, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	Functor src(ctx, { {"vec", &vec}, {"pred", &pred} }, { "id" }, "        return pred(vec[id]);\n");
	return general_find(ctx, begin, end, src, result);
}

bool TRTC_Find_If_Not(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, size_t& result, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	Functor src(ctx, { {"vec", &vec}, {"pred", &pred} }, { "id" }, "        return !pred(vec[id]);\n");
	return general_find(ctx, begin, end, src, result);
}
