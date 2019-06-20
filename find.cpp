#include "find.h"
#include "general_find.h"

bool TRTC_Find(const DVVectorLike& vec, const DeviceViewable& value, size_t& result)
{
	Functor src({ {"vec", &vec}, {"value", &value} }, { "id" }, "        return vec[id]==value;\n");
	return general_find(vec.size(), src, result);
}

bool TRTC_Find_If(const DVVectorLike& vec, const Functor& pred, size_t& result)
{
	Functor src({ {"vec", &vec}, {"pred", &pred} }, { "id" }, "        return pred(vec[id]);\n");
	return general_find(vec.size(), src, result);
}

bool TRTC_Find_If_Not(const DVVectorLike& vec, const Functor& pred, size_t& result)
{
	Functor src({ {"vec", &vec}, {"pred", &pred} }, { "id" }, "        return !pred(vec[id]);\n");
	return general_find(vec.size(), src, result);
}
