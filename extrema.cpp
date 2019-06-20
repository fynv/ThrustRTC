#include "extrema.h"
#include "general_reduce.h"
#include "built_in.h"

bool TRTC_Min_Element(const DVVectorLike& vec, size_t& id_min)
{
	Functor op({ {"vec", &vec} }, { "i1", "i2" }, "        return vec[i2]<vec[i1]?i2:i1;\n");
	if (vec.size() < 1) return true;
	ViewBuf buf;
	if (!general_reduce(vec.size(), "size_t", Functor("Identity"), op, buf)) return false;
	id_min = *(size_t*)buf.data();
	return true;
}

bool TRTC_Min_Element(const DVVectorLike& vec, const Functor& comp, size_t& id_min)
{
	Functor op({ {"vec", &vec}, {"comp", &comp} }, { "i1", "i2" }, "        return comp(vec[i2], vec[i1])?i2:i1; \n");
	if (vec.size() < 1) return true;
	ViewBuf buf;
	if (!general_reduce(vec.size(), "size_t", Functor("Identity"), op, buf)) return false;
	id_min = *(size_t*)buf.data();
	return true;
}


bool TRTC_Max_Element(const DVVectorLike& vec, size_t& id_max)
{
	Functor op({ {"vec", &vec} }, { "i1", "i2" }, "        return vec[i1]<vec[i2]?i2:i1;\n");
	if (vec.size() < 1) return true;
	ViewBuf buf;
	if (!general_reduce(vec.size(), "size_t", Functor("Identity"), op, buf)) return false;
	id_max = *(size_t*)buf.data();
	return true;
}


bool TRTC_Max_Element(const DVVectorLike& vec, const Functor& comp, size_t& id_max)
{
	Functor op({ {"vec", &vec}, {"comp", &comp} }, { "i1", "i2" }, "        return comp(vec[i1], vec[i2])?i2:i1; \n");
	if (vec.size() < 1) return true;
	ViewBuf buf;
	if (!general_reduce(vec.size(), "size_t", Functor("Identity"), op, buf)) return false;
	id_max = *(size_t*)buf.data();
	return true;
}

bool TRTC_MinMax_Element(const DVVectorLike& vec, size_t& id_min, size_t& id_max)
{
	Functor src({}, { "idx" }, 
		"        return Pair<size_t, size_t>({idx, idx});\n");
	Functor op( { {"vec", &vec} }, { "i1", "i2" }, 
		"        return Pair<size_t, size_t>({vec[i2.first]<vec[i1.first]?i2.first:i1.first, vec[i1.second]<vec[i2.second]?i2.second:i1.second });\n");

	if (vec.size() < 1) return true;

	ViewBuf buf;
	if (!general_reduce(vec.size(), "Pair<size_t, size_t>", src, op, buf)) return false;
	Pair<size_t, size_t> res = *(Pair<size_t, size_t>*)buf.data();
	id_min = res.first;
	id_max = res.second;
	return true;
}

bool TRTC_MinMax_Element(const DVVectorLike& vec, const Functor& comp, size_t& id_min, size_t& id_max)
{
	Functor src( { }, { "idx" },
		"        return Pair<size_t, size_t>({idx, idx});\n");

	Functor op( { {"vec", &vec}, {"comp", &comp} }, { "i1", "i2" }, 
		"        return Pair<size_t, size_t>({ comp(vec[i2.first],vec[i1.first])?i2.first:i1.first, comp(vec[i1.second],vec[i2.second])?i2.second:i1.second });\n");

	if (vec.size() < 1) return true;

	ViewBuf buf;
	if (!general_reduce(vec.size(), "Pair<size_t, size_t>", src, op, buf)) return false;
	Pair<size_t, size_t> res = *(Pair<size_t, size_t>*)buf.data();
	id_min = res.first;
	id_max = res.second;
	return true;
}

