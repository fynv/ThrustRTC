#include "sequence.h"

bool TRTC_Sequence(TRTCContext& ctx, DVVectorLike& vec, size_t begin, size_t end)
{
	static TRTC_For s_for( { "view_vec" }, "idx",
	"    view_vec[idx]= (decltype(view_vec)::value_t)(idx-_begin);\n" );

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec };
	return s_for.launch(ctx, begin, end, args);
}

bool TRTC_Sequence(TRTCContext& ctx, DVVectorLike& vec, const DeviceViewable& value_init, size_t begin, size_t end)
{
	static TRTC_For s_for(
	{ "view_vec", "value_init" }, "idx",
	"    view_vec[idx]= (decltype(view_vec)::value_t)value_init + (decltype(view_vec)::value_t)(idx-_begin);\n"
	);

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &value_init };
	s_for.launch(ctx, begin, end, args);
	return true;
}

bool THRUST_RTC_API TRTC_Sequence(TRTCContext& ctx, DVVectorLike& vec, const DeviceViewable& value_init, const DeviceViewable& value_step, size_t begin, size_t end)
{
	static TRTC_For s_for(
	{  "view_vec", "value_init", "value_step" }, "idx",
	"    view_vec[idx]= (decltype(view_vec)::value_t)value_init + (decltype(view_vec)::value_t)(idx-_begin)*(decltype(view_vec)::value_t)value_step;\n"
	);

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &value_init, &value_step };
	s_for.launch(ctx, begin, end, args);
	return true;
}
