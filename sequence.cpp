#include "sequence.h"

bool TRTC_Sequence(DVVectorLike& vec, size_t begin, size_t end)
{
	static TRTC_For s_for( { "view_vec", "begin" }, "idx",
	"    view_vec[idx + begin]= (decltype(view_vec)::value_t)idx;\n" );

	if (end == (size_t)(-1)) end = vec.size();
	DVSizeT dvbegin(begin);
	const DeviceViewable* args[] = { &vec, &dvbegin };
	return s_for.launch_n(end - begin, args);
}

bool TRTC_Sequence(DVVectorLike& vec, const DeviceViewable& value_init, size_t begin, size_t end)
{
	static TRTC_For s_for(
	{ "view_vec",  "begin", "value_init" }, "idx",
	"    view_vec[idx + begin]= (decltype(view_vec)::value_t)value_init + (decltype(view_vec)::value_t)idx;\n"
	);

	if (end == (size_t)(-1)) end = vec.size();
	DVSizeT dvbegin(begin);
	const DeviceViewable* args[] = { &vec, &dvbegin, &value_init };
	s_for.launch_n(end - begin, args);
	return true;
}

bool TRTC_Sequence(DVVectorLike& vec, const DeviceViewable& value_init, const DeviceViewable& value_step, size_t begin, size_t end)
{
	static TRTC_For s_for(
	{  "view_vec", "begin", "value_init", "value_step" }, "idx",
	"    view_vec[idx + begin]= (decltype(view_vec)::value_t)value_init + (decltype(view_vec)::value_t)idx*(decltype(view_vec)::value_t)value_step;\n"
	);

	if (end == (size_t)(-1)) end = vec.size();
	DVSizeT dvbegin(begin);
	const DeviceViewable* args[] = { &vec, &dvbegin, &value_init, &value_step };
	s_for.launch_n(end - begin, args);
	return true;
}
