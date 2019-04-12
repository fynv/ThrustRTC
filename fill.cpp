#include "fill.h"

bool TRTC_Fill(TRTCContext& ctx, DVVector& vec, const DeviceViewable& value, size_t begin, size_t end)
{
	static TRTC_For s_for(
	{ "view_vec", "value"  }, "idx",
	"    view_vec[idx]=value;"
	);

	if (vec.name_elem_cls() != value.name_view_cls())
	{
		printf("TRTC_Fill: vector type mismatch with value type.\n");
		return false;
	}

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &value };
	s_for.launch(ctx, begin, end, args);
	return true;
}
