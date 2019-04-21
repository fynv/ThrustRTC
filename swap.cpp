#include "swap.h"

bool TRTC_Swap(TRTCContext& ctx, DVVectorLike& vec1, DVVectorLike& vec2, size_t begin1, size_t end1, size_t begin2)
{
	static TRTC_For s_for(
		{ "view_vec1", "view_vec2", "delta" }, "idx",
		"    decltype(view_vec1)::value_t t = view_vec1[idx];\n"
		"    view_vec1[idx] = (decltype(view_vec1)::value_t)view_vec2[idx + delta];\n "
		"    view_vec2[idx + delta]=(decltype(view_vec2)::value_t)t;\n"
	);

	if (end1 == (size_t)(-1)) end1 = vec1.size();
	DVInt32 dvdelta((int)begin2 - (int)begin1);
	const DeviceViewable* args[] = { &vec1, &vec2, &dvdelta };
	return s_for.launch(ctx, begin1, end1, args);
}
