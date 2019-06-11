#include "equal.h"

bool TRTC_Equal(const DVVectorLike& vec1, const DVVectorLike& vec2, bool& ret, size_t begin1, size_t end1, size_t begin2)
{
	static TRTC_For s_for({ "view_vec1", "view_vec2", "view_res", "begin1", "begin2" }, "idx",
		"    if (view_vec1[idx + begin1]!=(decltype(view_vec1)::value_t)view_vec2[idx + begin2]) view_res[0]=false;\n"
	);

	if (end1 == (size_t)(-1)) end1 = vec1.size();
	
	ret = true;
	DVVector dvres("bool", 1, &ret);
	DVSizeT dvbegin1(begin1);
	DVSizeT dvbegin2(begin2);

	const DeviceViewable* args[] = { &vec1, &vec2, &dvres, &dvbegin1, &dvbegin2 };
	if (!s_for.launch_n(end1 - begin1, args)) return false;
	dvres.to_host(&ret);
	return true;
}

bool TRTC_Equal(const DVVectorLike& vec1, const DVVectorLike& vec2, const Functor& binary_pred, bool& ret, size_t begin1, size_t end1, size_t begin2)
{
	static TRTC_For s_for({ "view_vec1", "view_vec2", "view_res", "binary_pred", "begin1", "begin2" }, "idx",
		"    if (!binary_pred(view_vec1[idx + begin1], view_vec2[idx + begin2])) view_res[0]=false;\n"
	);

	if (end1 == (size_t)(-1)) end1 = vec1.size();

	ret = true;
	DVVector dvres("bool", 1, &ret);
	DVSizeT dvbegin1(begin1);
	DVSizeT dvbegin2(begin2);

	const DeviceViewable* args[] = { &vec1, &vec2, &dvres, &binary_pred, &dvbegin1, &dvbegin2 };
	if (!s_for.launch_n(end1 - begin1, args)) return false;
	dvres.to_host(&ret);
	return true;
}
