#include "swap.h"

bool TRTC_Swap(DVVectorLike& vec1, DVVectorLike& vec2)
{
	static TRTC_For s_for(
		{ "view_vec1", "view_vec2" }, "idx",
		"    decltype(view_vec1)::value_t t = view_vec1[idx];\n"
		"    view_vec1[idx] = (decltype(view_vec1)::value_t)view_vec2[idx];\n "
		"    view_vec2[idx]=(decltype(view_vec2)::value_t)t;\n"
	);

	const DeviceViewable* args[] = { &vec1, &vec2 };
	return s_for.launch_n( vec1.size(), args);
}
