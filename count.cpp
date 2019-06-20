#include "count.h"
#include "general_reduce.h"
#include "built_in.h"

bool TRTC_Count(const DVVectorLike& vec, const DeviceViewable& value, size_t& ret)
{
	Functor src({ {"vec_in", &vec}, {"eq_value", &value } }, { "idx" },
		"        return (vec_in[idx] == (decltype(vec_in)::value_t)eq_value)?1:0;\n");

	Functor op("Plus");

	ret = 0;
	if (vec.size() < 1) return true;

	ViewBuf buf;
	if (!general_reduce(vec.size(), "size_t", src, op, buf)) return false;
	ret = *(size_t*)buf.data();
	return true;
}

bool TRTC_Count_If(const DVVectorLike& vec, const Functor& pred, size_t& ret)
{
	Functor src({ {"vec_in", &vec}, {"pred", &pred } }, { "idx" },
		"        return pred(vec_in[idx])?1:0;\n");
	Functor op("Plus");

	ret = 0;
	if (vec.size() < 1) return true;

	ViewBuf buf;
	if (!general_reduce(vec.size(), "size_t", src, op, buf)) return false;
	ret = *(size_t*)buf.data();
	return true;

}