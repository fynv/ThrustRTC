#include "extrema.h"
#include "general_reduce.h"

bool TRTC_Min_Element(TRTCContext& ctx, const DVVectorLike& vec, size_t& id_min, size_t begin, size_t end)
{
	DVSizeT dvbegin(begin);
	
	Functor src(ctx, { {"begin", &dvbegin } }, { "idx" }, "        return begin + idx;\n");
	Functor op(ctx, { {"vec", &vec} }, { "i1", "i2" }, "        return vec[i2]<vec[i1]?i2:i1;\n");

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

	Functor src(ctx, { {"begin", &dvbegin } }, { "idx" }, "        return begin + idx;\n");
	Functor op(ctx, { {"vec", &vec}, {"comp", &comp} }, { "i1", "i2" }, "        return comp(vec[i2], vec[i1])?i2:i1; \n");

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
	
	Functor src(ctx, { {"begin", &dvbegin } }, { "idx" }, "        return begin + idx;\n");
	Functor op(ctx, { {"vec", &vec} }, { "i1", "i2" }, "        return vec[i1]<vec[i2]?i2:i1;\n");

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

	Functor src(ctx, { {"begin", &dvbegin } }, { "idx" }, "        return begin + idx;\n");
	Functor op(ctx, { {"vec", &vec}, {"comp", &comp} }, { "i1", "i2" }, "        return comp(vec[i1], vec[i2])?i2:i1; \n");

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

	std::string d_MinMaxIds = ctx.add_struct(
		"    size_t id_min;\n"
		"    size_t id_max;\n"
	);

	DVSizeT dvbegin(begin);

	Functor src(ctx, { {"begin", &dvbegin } }, { "idx" }, 
		(std::string("        return ")+ d_MinMaxIds+"({begin + idx, begin + idx});\n").c_str());
	Functor op(ctx, { {"vec", &vec} }, { "i1", "i2" }, 
		(std::string("        return ") + d_MinMaxIds+"({vec[i2.id_min]<vec[i1.id_min]?i2.id_min:i1.id_min, vec[i1.id_max]<vec[i2.id_max]?i2.id_max:i1.id_max });\n").c_str());

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

	std::string d_MinMaxIds = ctx.add_struct(
		"    size_t id_min;\n"
		"    size_t id_max;\n"
	);

	DVSizeT dvbegin(begin);
	Functor src(ctx, { {"begin", &dvbegin } }, { "idx" },
		(std::string("        return ") + d_MinMaxIds + "({begin + idx, begin + idx});\n").c_str());

	Functor op(ctx, { {"vec", &vec}, {"comp", &comp} }, { "i1", "i2" }, 
		(std::string("        return ") + d_MinMaxIds + "({ comp(vec[i2.id_min],vec[i1.id_min])?i2.id_min:i1.id_min, comp(vec[i1.id_max],vec[i2.id_max])?i2.id_max:i1.id_max });\n").c_str());

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;

	ViewBuf buf;
	if (!general_reduce(ctx, end - begin, d_MinMaxIds.c_str(), src, op, buf)) return false;
	MinMaxIds res = *(MinMaxIds*)buf.data();
	id_min = res.id_min;
	id_max = res.id_max;
	return true;
}

