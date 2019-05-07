#include "logical.h"

bool TRTC_All_Of(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, bool& ret, size_t begin, size_t end)
{
	static TRTC_For s_for({ "view_vec", "view_res", "pred" }, "idx",
		"    if (!pred(view_vec[idx])) view_res[0]=false;\n"
	);
	if (end == (size_t)(-1)) end = vec.size();
	ret = false;
	if (end - begin < 1) return true;
	ret = true;
	DVVector dvres(ctx, "bool", 1, &ret);
	const DeviceViewable* args[] = { &vec, &dvres, &pred };
	if (!s_for.launch(ctx, begin, end, args)) return false;
	dvres.to_host(&ret);
	return true;
}

bool TRTC_Any_Of(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, bool& ret, size_t begin, size_t end)
{
	static TRTC_For s_for({ "view_vec", "view_res", "pred" }, "idx",
		"    if (pred(view_vec[idx])) view_res[0]=true;\n"
	);

	if (end == (size_t)(-1)) end = vec.size();
	ret = false;
	if (end - begin < 1) return true;

	DVVector dvres(ctx, "bool", 1, &ret);
	const DeviceViewable* args[] = { &vec, &dvres, &pred };
	if (!s_for.launch(ctx, begin, end, args)) return false;
	dvres.to_host(&ret);
	return true;
}


bool TRTC_None_Of(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, bool& ret, size_t begin, size_t end)
{
	static TRTC_For s_for({ "view_vec", "view_res", "pred" }, "idx",
		"    if (pred(view_vec[idx])) view_res[0]=false;\n"
	);

	if (end == (size_t)(-1)) end = vec.size();
	ret = true;
	if (end - begin < 1) return true;

	DVVector dvres(ctx, "bool", 1, &ret);
	const DeviceViewable* args[] = { &vec, &dvres, &pred };
	if (!s_for.launch(ctx, begin, end, args)) return false;
	dvres.to_host(&ret);
	return true;
}

