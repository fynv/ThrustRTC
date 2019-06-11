#include "extrema.h"
#include "general_reduce.h"
#include "built_in.h"

bool TRTC_Min_Element(const DVVectorLike& vec, size_t& id_min, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);
	
	Functor src({ {"begin", &dvbegin } }, { "idx" }, "        return begin + idx;\n");
	Functor op({ {"vec", &vec} }, { "i1", "i2" }, "        return vec[i2]<vec[i1]?i2:i1;\n");

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(end - begin, "size_t", src, op, buf)) return false;
	id_min = *(size_t*)buf.data();
	return true;
}

bool TRTC_Min_Element(const DVVectorLike& vec, const Functor& comp, size_t& id_min, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);

	Functor src({ {"begin", &dvbegin } }, { "idx" }, "        return begin + idx;\n");
	Functor op({ {"vec", &vec}, {"comp", &comp} }, { "i1", "i2" }, "        return comp(vec[i2], vec[i1])?i2:i1; \n");

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(end - begin, "size_t", src, op, buf)) return false;
	id_min = *(size_t*)buf.data();
	return true;
}


bool TRTC_Max_Element(const DVVectorLike& vec, size_t& id_max, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);
	
	Functor src({ {"begin", &dvbegin } }, { "idx" }, "        return begin + idx;\n");
	Functor op({ {"vec", &vec} }, { "i1", "i2" }, "        return vec[i1]<vec[i2]?i2:i1;\n");

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(end - begin, "size_t", src, op, buf)) return false;
	id_max = *(size_t*)buf.data();
	return true;
}


bool TRTC_Max_Element(const DVVectorLike& vec, const Functor& comp, size_t& id_max, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);

	Functor src({ {"begin", &dvbegin } }, { "idx" }, "        return begin + idx;\n");
	Functor op({ {"vec", &vec}, {"comp", &comp} }, { "i1", "i2" }, "        return comp(vec[i1], vec[i2])?i2:i1; \n");

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(end - begin, "size_t", src, op, buf)) return false;
	id_max = *(size_t*)buf.data();
	return true;
}

bool TRTC_MinMax_Element(const DVVectorLike& vec, size_t& id_min, size_t& id_max, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);

	Functor src({ {"begin", &dvbegin } }, { "idx" }, 
		"        return Pair<size_t, size_t>({begin + idx, begin + idx});\n");
	Functor op( { {"vec", &vec} }, { "i1", "i2" }, 
		"        return Pair<size_t, size_t>({vec[i2.first]<vec[i1.first]?i2.first:i1.first, vec[i1.second]<vec[i2.second]?i2.second:i1.second });\n");

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce( end - begin, "Pair<size_t, size_t>", src, op, buf)) return false;
	Pair<size_t, size_t> res = *(Pair<size_t, size_t>*)buf.data();
	id_min = res.first;
	id_max = res.second;
	return true;
}

bool TRTC_MinMax_Element(const DVVectorLike& vec, const Functor& comp, size_t& id_min, size_t& id_max, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);
	Functor src( { {"begin", &dvbegin } }, { "idx" },
		"        return Pair<size_t, size_t>({begin + idx, begin + idx});\n");

	Functor op( { {"vec", &vec}, {"comp", &comp} }, { "i1", "i2" }, 
		"        return Pair<size_t, size_t>({ comp(vec[i2.frist],vec[i1.frist])?i2.frist:i1.frist, comp(vec[i1.second],vec[i2.second])?i2.second:i1.second });\n");

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce( end - begin, "Pair<size_t, size_t>", src, op, buf)) return false;
	Pair<size_t, size_t> res = *(Pair<size_t, size_t>*)buf.data();
	id_min = res.first;
	id_max = res.second;
	return true;
}

