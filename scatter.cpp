#include "scatter.h"

bool TRTC_Scatter(TRTCContext& ctx, const DVVectorLike& vec_in, const DVVectorLike& vec_map, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_map", "view_vec_out", "delta_map", "delta_out" }, "idx",
		"    view_vec_out[view_vec_map[idx+delta_map]+delta_out] = (decltype(view_vec_out)::value_t)view_vec_in[idx];\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta_map((int)begin_map - (int)begin_in);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);

	const DeviceViewable* args[] = { &vec_in, &vec_map, &vec_out, &dvdelta_map, &dvdelta_out };
	return s_for.launch(ctx, begin_in, end_in, args);
}

bool TRTC_Scatter_If(TRTCContext& ctx, const DVVectorLike& vec_in, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_stencil, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_map", "view_vec_stencil", "view_vec_out", "delta_map", "delta_stencil", "delta_out" }, "idx",
		"    if(view_vec_stencil[idx+delta_stencil])\n"
		"        view_vec_out[view_vec_map[idx+delta_map]+delta_out] = (decltype(view_vec_out)::value_t)view_vec_in[idx];\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta_map((int)begin_map - (int)begin_in);
	DVInt32 dvdelta_stencil((int)begin_stencil - (int)begin_in);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_map, &vec_stencil, &vec_out, &dvdelta_map, &dvdelta_stencil, &dvdelta_out };
	return s_for.launch(ctx, begin_in, end_in, args);
}

bool TRTC_Scatter_If(TRTCContext& ctx, const DVVectorLike& vec_in, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_stencil, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_map", "view_vec_stencil", "view_vec_out", "pred", "delta_map", "delta_stencil", "delta_out" }, "idx",
		"    if(pred(view_vec_stencil[idx+delta_stencil]))\n"
		"        view_vec_out[view_vec_map[idx+delta_map]+delta_out] = (decltype(view_vec_out)::value_t)view_vec_in[idx];\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta_map((int)begin_map - (int)begin_in);
	DVInt32 dvdelta_stencil((int)begin_stencil - (int)begin_in);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_map, &vec_stencil, &vec_out, &pred, &dvdelta_map, &dvdelta_stencil, &dvdelta_out };
	return s_for.launch(ctx, begin_in, end_in, args);
}

