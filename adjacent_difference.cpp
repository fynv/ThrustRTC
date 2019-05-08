#include "adjacent_difference.h"

bool TRTC_Adjacent_Difference(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "begin_in", "begin_out"}, "idx",
	"    auto value = view_vec_in[idx + begin_in];\n"
	"    if (idx>0) value -= view_vec_in[idx - 1 + begin_in]; \n"
	"    view_vec_out[idx + begin_out] = (decltype(view_vec_out)::value_t) value;\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &dvbegin_in, &dvbegin_out };
	return s_for.launch_n(ctx, end_in - begin_in, args);
}

bool TRTC_Adjacent_Difference(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_op, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "binary_op", "begin_in", "begin_out" }, "idx",
		"    auto value = view_vec_in[idx + begin_in];\n"
		"    if (idx>0) value = (decltype(view_vec_in)::value_t) binary_op(value, view_vec_in[idx - 1 + begin_in]); \n"
		"    view_vec_out[idx + begin_out] = (decltype(view_vec_out)::value_t) value;\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &binary_op, &dvbegin_in, &dvbegin_out };
	return s_for.launch_n(ctx, end_in - begin_in, args);
}
