#include "equal.h"

bool TRTC_Equal(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, bool& ret, size_t begin1, size_t end1, size_t begin2)
{
	static TRTC_For s_for({ "view_vec1", "view_vec2", "view_res", "delta2" }, "idx",
		"    if (view_vec1[idx]!=(decltype(view_vec1)::value_t)view_vec2[idx + delta2]) view_res[0]=false;\n"
	);

	if (end1 == (size_t)(-1)) end1 = vec1.size();
	
	ret = true;
	DVVector dvres(ctx, "bool", 1, &ret);
	DVInt32 dvdelta2((int)begin2 - (int)begin1);

	const DeviceViewable* args[] = { &vec1, &vec2, &dvres, &dvdelta2 };
	if (!s_for.launch(ctx, begin1, end1, args)) return false;
	dvres.to_host(&ret);
	return true;
}

bool TRTC_Equal(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, const Functor& binary_pred, bool& ret, size_t begin1, size_t end1, size_t begin2)
{
	static TRTC_For s_for({ "view_vec1", "view_vec2", "view_res", "binary_pred", "delta2" }, "idx",
		"    if (!binary_pred(view_vec1[idx], view_vec2[idx + delta2])) view_res[0]=false;\n"
	);

	if (end1 == (size_t)(-1)) end1 = vec1.size();

	ret = true;
	DVVector dvres(ctx, "bool", 1, &ret);
	DVInt32 dvdelta2((int)begin2 - (int)begin1);

	const DeviceViewable* args[] = { &vec1, &vec2, &dvres, &binary_pred, &dvdelta2 };
	if (!s_for.launch(ctx, begin1, end1, args)) return false;
	dvres.to_host(&ret);
	return true;
}