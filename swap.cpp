#include "swap.h"

bool TRTC_Swap(TRTCContext& ctx, DVVectorLike& vec1, DVVectorLike& vec2, size_t begin1, size_t end1, size_t begin2)
{
	static TRTC_For s_for(
		{ "view_vec1", "view_vec2", "begin1", "begin2" }, "idx",
		"    decltype(view_vec1)::value_t t = view_vec1[idx + begin1];\n"
		"    view_vec1[idx + begin1] = (decltype(view_vec1)::value_t)view_vec2[idx + begin2];\n "
		"    view_vec2[idx + begin2]=(decltype(view_vec2)::value_t)t;\n"
	);

	if (end1 == (size_t)(-1)) end1 = vec1.size();
	DVSizeT dvbegin1(begin1);
	DVSizeT dvbegin2(begin2);
	const DeviceViewable* args[] = { &vec1, &vec2, &dvbegin1, &dvbegin2 };
	return s_for.launch_n(ctx, end1-begin1, args);
}
