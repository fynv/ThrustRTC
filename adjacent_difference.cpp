#include "adjacent_difference.h"

bool TRTC_Adjacent_Difference(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "delta"}, "idx",
	"    auto value = view_vec_in[idx];\n"
	"    if (idx>0) value -= view_vec_in[idx-1]; \n"
	"    view_vec_out[idx+delta] = (decltype(view_vec_out)::value_t) value;\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &dvdelta };
	return s_for.launch(ctx, begin_in, end_in, args);
}

bool TRTC_Adjacent_Difference(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_op, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "binary_op", "delta" }, "idx",
		"    auto value = view_vec_in[idx];\n"
		"    if (idx>0) value = (decltype(view_vec_in)::value_t) binary_op(value, view_vec_in[idx-1]); \n"
		"    view_vec_out[idx+delta] = (decltype(view_vec_out)::value_t) value;\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &binary_op, &dvdelta };
	return s_for.launch(ctx, begin_in, end_in, args);
}
