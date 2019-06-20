#include "adjacent_difference.h"

bool TRTC_Adjacent_Difference(const DVVectorLike& vec_in, DVVectorLike& vec_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out"}, "idx",
	"    auto value = view_vec_in[idx ];\n"
	"    if (idx>0) value -= view_vec_in[idx - 1]; \n"
	"    view_vec_out[idx] = (decltype(view_vec_out)::value_t) value;\n"
	);

	const DeviceViewable* args[] = { &vec_in, &vec_out };
	return s_for.launch_n(vec_in.size(), args);
}

bool TRTC_Adjacent_Difference(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_op)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "binary_op" }, "idx",
		"    auto value = view_vec_in[idx];\n"
		"    if (idx>0) value = (decltype(view_vec_in)::value_t) binary_op(value, view_vec_in[idx - 1]); \n"
		"    view_vec_out[idx] = (decltype(view_vec_out)::value_t) value;\n"
	);

	const DeviceViewable* args[] = { &vec_in, &vec_out, &binary_op };
	return s_for.launch_n(vec_in.size(), args);
}
