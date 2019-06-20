#include "replace.h"

bool TRTC_Replace(DVVectorLike& vec, const DeviceViewable& old_value, const DeviceViewable& new_value)
{
	static TRTC_For s_for( {"view_vec", "old_value", "new_value" }, "idx",
		"    if (view_vec[idx]==(decltype(view_vec)::value_t)old_value) view_vec[idx] = (decltype(view_vec)::value_t)new_value;\n"
	);

	const DeviceViewable* args[] = { &vec, &old_value, &new_value };
	return s_for.launch_n(vec.size(), args);
}

bool TRTC_Replace_If(DVVectorLike& vec, const Functor& pred, const DeviceViewable& new_value)
{
	static TRTC_For s_for({ "view_vec", "pred", "new_value"}, "idx",
		"    if (pred(view_vec[idx])) view_vec[idx] = (decltype(view_vec)::value_t)new_value;\n"
	);

	const DeviceViewable* args[] = { &vec, &pred, &new_value };
	return s_for.launch_n(vec.size(), args);
}

bool TRTC_Replace_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& old_value, const DeviceViewable& new_value)
{
	static TRTC_For s_for(
	{ "view_vec_in", "view_vec_out" , "old_value", "new_value" }, "idx",
	"    auto value = view_vec_in[idx];\n"
	"    view_vec_out[idx] = value == (decltype(view_vec_in)::value_t)old_value ?  (decltype(view_vec_out)::value_t)new_value :  (decltype(view_vec_out)::value_t)value;\n"
	);

	const DeviceViewable* args[] = { &vec_in, &vec_out, &old_value, &new_value };
	return s_for.launch_n(vec_in.size(), args);
}

bool TRTC_Replace_Copy_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred, const DeviceViewable& new_value)
{
	static TRTC_For s_for(
		{ "view_vec_in", "view_vec_out" , "pred", "new_value" }, "idx",
		"    auto value = view_vec_in[idx];\n"
		"    view_vec_out[idx] = pred(value) ?  (decltype(view_vec_out)::value_t)new_value :  (decltype(view_vec_out)::value_t)value;\n"
	);
	
	const DeviceViewable* args[] = { &vec_in, &vec_out, &pred, &new_value };
	return s_for.launch_n(vec_in.size(), args);
}
