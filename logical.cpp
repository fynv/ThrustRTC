#include "logical.h"

bool TRTC_All_Of(const DVVectorLike& vec, const Functor& pred, bool& ret)
{
	static TRTC_For s_for({ "view_vec", "view_res", "pred" }, "idx",
		"    if (!pred(view_vec[idx])) view_res[0]=false;\n"
	);
	ret = false;
	if (vec.size() < 1) return true;
	ret = true;
	DVVector dvres("bool", 1, &ret);
	const DeviceViewable* args[] = { &vec, &dvres, &pred };
	if (!s_for.launch_n(vec.size(), args)) return false;
	dvres.to_host(&ret);
	return true;
}

bool TRTC_Any_Of(const DVVectorLike& vec, const Functor& pred, bool& ret)
{
	static TRTC_For s_for({ "view_vec", "view_res", "pred" }, "idx",
		"    if (pred(view_vec[idx])) view_res[0]=true;\n"
	);

	ret = false;
	if (vec.size() < 1) return true;

	DVVector dvres("bool", 1, &ret);
	const DeviceViewable* args[] = { &vec, &dvres, &pred };
	if (!s_for.launch_n(vec.size(), args)) return false;
	dvres.to_host(&ret);
	return true;
}


bool TRTC_None_Of(const DVVectorLike& vec, const Functor& pred, bool& ret)
{
	static TRTC_For s_for({ "view_vec", "view_res", "pred" }, "idx",
		"    if (pred(view_vec[idx])) view_res[0]=false;\n"
	);

	ret = true;
	if (vec.size() < 1) return true;

	DVVector dvres("bool", 1, &ret);
	const DeviceViewable* args[] = { &vec, &dvres, &pred };
	if (!s_for.launch_n(vec.size(), args)) return false;
	dvres.to_host(&ret);
	return true;
}

