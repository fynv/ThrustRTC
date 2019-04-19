#include "fill.h"

bool TRTC_Fill(TRTCContext& ctx, DVVector& vec, const DeviceViewable& value, size_t begin, size_t end)
{
	static TRTC_For s_for(
	{ "view_vec", "value"  }, "idx",
	"    view_vec[idx]=(decltype(view_vec)::value_t)value;"
	);

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &value };
	return s_for.launch(ctx, begin, end, args);
}
