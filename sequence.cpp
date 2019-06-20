#include "sequence.h"

bool TRTC_Sequence(DVVectorLike& vec)
{
	static TRTC_For s_for( { "view_vec" }, "idx",
	"    view_vec[idx]= (decltype(view_vec)::value_t)idx;\n" );

	const DeviceViewable* args[] = { &vec };
	return s_for.launch_n(vec.size(), args);
}

bool TRTC_Sequence(DVVectorLike& vec, const DeviceViewable& value_init)
{
	static TRTC_For s_for(
	{ "view_vec",  "value_init" }, "idx",
	"    view_vec[idx]= (decltype(view_vec)::value_t)value_init + (decltype(view_vec)::value_t)idx;\n"
	);

	const DeviceViewable* args[] = { &vec, &value_init };
	s_for.launch_n(vec.size(), args);
	return true;
}

bool TRTC_Sequence(DVVectorLike& vec, const DeviceViewable& value_init, const DeviceViewable& value_step)
{
	static TRTC_For s_for(
	{  "view_vec", "value_init", "value_step" }, "idx",
	"    view_vec[idx]= (decltype(view_vec)::value_t)value_init + (decltype(view_vec)::value_t)idx*(decltype(view_vec)::value_t)value_step;\n"
	);

	const DeviceViewable* args[] = { &vec, &value_init, &value_step };
	s_for.launch_n(vec.size(), args);
	return true;
}
