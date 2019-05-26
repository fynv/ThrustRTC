#include "mismatch.h"
#include "general_find.h"

bool TRTC_Mismatch(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, size_t& result1, size_t& result2, size_t begin1, size_t end1, size_t begin2)
{
	if (end1 == (size_t)(-1)) end1 = vec1.size();
	DVSizeT dvbegin1(begin1);
	DVSizeT dvbegin2(begin2);
	Functor src(ctx, { {"vec1", &vec1},  {"vec2", &vec2}, {"begin1", &dvbegin1}, {"begin2", &dvbegin2} }, { "id" }, "        return vec1[id+begin1]!=vec2[id+begin2];\n");
	size_t result;
	if (!general_find(ctx, 0, end1 - begin1, src, result)) return false;
	result1 = result + begin1;
	result2 = result + begin2;
	return true;
}

bool TRTC_Mismatch(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, const Functor& pred, size_t& result1, size_t& result2, size_t begin1, size_t end1, size_t begin2)
{
	if (end1 == (size_t)(-1)) end1 = vec1.size();
	DVSizeT dvbegin1(begin1);
	DVSizeT dvbegin2(begin2);
	Functor src(ctx, { {"vec1", &vec1}, {"vec2", &vec2}, {"pred", &pred}, {"begin1", &dvbegin1}, {"begin2", &dvbegin2} }, { "id" }, "        return !pred(vec1[id+begin1],vec2[id+begin2]);\n");
	size_t result;
	if (!general_find(ctx, 0, end1 - begin1, src, result)) return false;
	result1 = result + begin1;
	result2 = result + begin2;
	return true;
}


