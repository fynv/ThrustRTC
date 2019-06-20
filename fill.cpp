#include "fill.h"

bool TRTC_Fill(DVVectorLike& vec, const DeviceViewable& value)
{
	static TRTC_For s_for(
	{ "view_vec", "value"  }, "idx",
	"    view_vec[idx]=(decltype(view_vec)::value_t)value;"
	);

	const DeviceViewable* args[] = { &vec, &value };
	return s_for.launch_n(vec.size(), args);
}
