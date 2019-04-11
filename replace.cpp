#include "replace.h"
#include "for.h"

bool TRTC_Replace(TRTCContext& ctx, DVVector& vec, const DeviceViewable& old_value, const DeviceViewable& new_value, size_t begin, size_t end)
{
	static TRTC_For_Template s_templ(
	{ "T" },
	{ { "VectorView<T>", "view_vec" }, { "T", "old_value" },{ "T", "new_value" } }, "idx",
		"    if (view_vec[idx]==old_value) view_vec[idx] = new_value;"
	);

	if (vec.name_elem_cls() != old_value.name_view_cls())
	{
		printf("TRTC_Fill: vector type mismatch with old_value type.\n");
		return false;
	}

	if (vec.name_elem_cls() != new_value.name_view_cls())
	{
		printf("TRTC_Fill: vector type mismatch with new_value type.\n");
		return false;
	}

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &old_value, &new_value };
	s_templ.launch(ctx, begin, end, args);
	return true;

}
