#include "mismatch.h"
#include "general_find.h"

bool TRTC_Mismatch(const DVVectorLike& vec1, const DVVectorLike& vec2, size_t& result)
{
	Functor src({ {"vec1", &vec1},  {"vec2", &vec2}}, { "id" }, "        return vec1[id]!=vec2[id];\n");
	if (!general_find(vec1.size(), src, result)) return false;
	return true;
}

bool TRTC_Mismatch(const DVVectorLike& vec1, const DVVectorLike& vec2, const Functor& pred, size_t& result)
{
	Functor src({ {"vec1", &vec1}, {"vec2", &vec2}, {"pred", &pred} }, { "id" }, "        return !pred(vec1[id],vec2[id]);\n");
	if (!general_find(vec1.size(), src, result)) return false;
	return true;
}


