#include "copy.h"

bool TRTC_Copy(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for(
		{ "view_vec_in", "view_vec_out", "delta" }, "idx",
		"    view_vec_out[idx + delta]=(decltype(view_vec_out)::value_t)view_vec_in[idx];"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &dvdelta };
	return s_for.launch(ctx, begin_in, end_in, args);
}
