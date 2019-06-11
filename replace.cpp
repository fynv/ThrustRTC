#include "replace.h"

bool TRTC_Replace(DVVectorLike& vec, const DeviceViewable& old_value, const DeviceViewable& new_value, size_t begin, size_t end)
{
	static TRTC_For s_for( {"view_vec", "old_value", "new_value" }, "idx",
		"    if (view_vec[idx]==(decltype(view_vec)::value_t)old_value) view_vec[idx] = (decltype(view_vec)::value_t)new_value;\n"
	);

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &old_value, &new_value };
	return s_for.launch(begin, end, args);
}

bool TRTC_Replace_If(DVVectorLike& vec, const Functor& pred, const DeviceViewable& new_value, size_t begin, size_t end)
{
	static TRTC_For s_for({ "view_vec", "pred", "new_value"}, "idx",
		"    if (pred(view_vec[idx])) view_vec[idx] = (decltype(view_vec)::value_t)new_value;\n"
	);

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &pred, &new_value };
	return s_for.launch(begin, end, args);
}

bool TRTC_Replace_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& old_value, const DeviceViewable& new_value, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for(
	{ "view_vec_in", "view_vec_out" , "old_value", "new_value", "begin_in", "begin_out" }, "idx",
	"    auto value = view_vec_in[idx + begin_in];\n"
	"    view_vec_out[idx + begin_out] = value == (decltype(view_vec_in)::value_t)old_value ?  (decltype(view_vec_out)::value_t)new_value :  (decltype(view_vec_out)::value_t)value;\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &old_value, &new_value,  &dvbegin_in, &dvbegin_out };
	return s_for.launch_n(end_in - begin_in, args);
}

bool TRTC_Replace_Copy_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred, const DeviceViewable& new_value, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for(
		{ "view_vec_in", "view_vec_out" , "pred", "new_value", "begin_in", "begin_out" }, "idx",
		"    auto value = view_vec_in[idx + begin_in];\n"
		"    view_vec_out[idx + begin_out] = pred(value) ?  (decltype(view_vec_out)::value_t)new_value :  (decltype(view_vec_out)::value_t)value;\n"
	);
	
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &pred, &new_value, &dvbegin_in, &dvbegin_out };
	return s_for.launch_n(end_in - begin_in, args);
}
