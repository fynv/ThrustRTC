#include "equal.h"

bool TRTC_Equal(const DVVectorLike& vec1, const DVVectorLike& vec2, bool& ret)
{
	static TRTC_For s_for({ "view_vec1", "view_vec2", "view_res" }, "idx",
		"    if (view_vec1[idx ]!=(decltype(view_vec1)::value_t)view_vec2[idx ]) view_res[0]=false;\n"
	);
	
	ret = true;
	DVVector dvres("bool", 1, &ret);

	const DeviceViewable* args[] = { &vec1, &vec2, &dvres };
	if (!s_for.launch_n(vec1.size(), args)) return false;
	dvres.to_host(&ret);
	return true;
}

bool TRTC_Equal(const DVVectorLike& vec1, const DVVectorLike& vec2, const Functor& binary_pred, bool& ret)
{
	static TRTC_For s_for({ "view_vec1", "view_vec2", "view_res", "binary_pred" }, "idx",
		"    if (!binary_pred(view_vec1[idx], view_vec2[idx])) view_res[0]=false;\n"
	);

	ret = true;
	DVVector dvres("bool", 1, &ret);
	
	const DeviceViewable* args[] = { &vec1, &vec2, &dvres, &binary_pred };
	if (!s_for.launch_n(vec1.size(), args)) return false;
	dvres.to_host(&ret);
	return true;
}
