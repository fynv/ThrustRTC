#include "sequence.h"
#include "for.h"

void TRTC_Sequence(TRTCContext& ctx, DVVector& vec, size_t begin, size_t end)
{
	static TRTC_For_Template s_templ(
	{ "T" },
	{ { "VectorView<T>", "view_vec" } }, "idx",
	"    view_vec[idx]= (T)(idx-_begin);"
	);

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec };
	s_templ.launch(ctx, begin, end, args);
}

bool TRTC_Sequence(TRTCContext& ctx, DVVector& vec, const DeviceViewable& value_init, size_t begin, size_t end)
{
	static TRTC_For_Template s_templ(
	{ "T" },
	{ { "VectorView<T>", "view_vec" }, { "T", "value_init" } }, "idx",
	"    view_vec[idx]= value_init + (T)(idx-_begin);"
	);

	if (vec.name_elem_cls() != value_init.name_view_cls())
	{
		printf("TRTC_Sequence: vector type mismatch with value_init type.\n");
		return false;
	}

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &value_init };
	s_templ.launch(ctx, begin, end, args);
	return true;
}

bool THRUST_RTC_API TRTC_Sequence(TRTCContext& ctx, DVVector& vec, const DeviceViewable& value_init, const DeviceViewable& value_step, size_t begin, size_t end)
{
	static TRTC_For_Template s_templ(
	{ "T" },
	{ { "VectorView<T>", "view_vec" }, { "T", "value_init" }, { "T", "value_step" } }, "idx",
	"    view_vec[idx]= value_init + (T)(idx-_begin)*value_step;"
	);

	if (vec.name_elem_cls() != value_init.name_view_cls())
	{
		printf("TRTC_Sequence: vector type mismatch with value_init type.\n");
		return false;
	}

	if (vec.name_elem_cls() != value_step.name_view_cls())
	{
		printf("TRTC_Sequence: vector type mismatch with value_step type.\n");
		return false;
	}

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &value_init, &value_step };
	s_templ.launch(ctx, begin, end, args);
	return true;
}
