#include "scatter.h"

bool TRTC_Scatter(TRTCContext& ctx, const DVVectorLike& vec_in, const DVVectorLike& vec_map, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_map", "view_vec_out", "delta_map", "delta_out" }, "idx",
		"    view_vec_out[view_vec_map[idx+delta_map]+delta_out] = (decltype(view_vec_out)::value_t)view_vec_in[idx];\n"
	);

	DVInt32 dvdelta_map((int)begin_map - (int)begin_in);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	const DeviceViewable* args[] = { &vec_in, &vec_map, &vec_out, &dvdelta_map, &dvdelta_out };
	return s_for.launch(ctx, begin_in, end_in, args);
}

bool TRTC_Scatter_If(TRTCContext& ctx, const DVVectorLike& vec_in, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_stencil, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_map", "view_vec_stencil", "view_vec_out", "delta_map", "delta_stencil", "delta_out" }, "idx",
		"    if(view_vec_stencil[idx+delta_stencil])\n"
		"        view_vec_out[view_vec_map[idx+delta_map]+delta_out] = (decltype(view_vec_out)::value_t)view_vec_in[idx];\n"
	);

	DVInt32 dvdelta_map((int)begin_map - (int)begin_in);
	DVInt32 dvdelta_stencil((int)begin_stencil - (int)begin_in);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	const DeviceViewable* args[] = { &vec_in, &vec_map, &vec_stencil, &vec_out, &dvdelta_map, &dvdelta_stencil, &dvdelta_out };
	return s_for.launch(ctx, begin_in, end_in, args);
}

bool TRTC_Scatter_If(TRTCContext& ctx, const DVVectorLike& vec_in, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_stencil, size_t begin_out)
{
	DVInt32 dvdelta_map((int)begin_map - (int)begin_in);
	DVInt32 dvdelta_stencil((int)begin_stencil - (int)begin_in);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);
	std::vector<TRTCContext::AssignedParam> arg_map = pred.arg_map;
	arg_map.push_back({ "_view_vec_in", &vec_in });
	arg_map.push_back({ "_view_vec_map", &vec_map });
	arg_map.push_back({ "_view_vec_stencil", &vec_stencil });
	arg_map.push_back({ "_view_vec_out", &vec_out });
	arg_map.push_back({ "_delta_map", &dvdelta_map });
	arg_map.push_back({ "_delta_stencil", &dvdelta_stencil });
	arg_map.push_back({ "_delta_out", &dvdelta_out });

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	return ctx.launch_for(begin_in, end_in, arg_map, "_idx",
		(pred.generate_code("bool", { "_view_vec_stencil[_idx+_delta_stencil]" }) +
			"    if(" + pred.functor_ret + ")\n"
			"        _view_vec_out[_view_vec_map[_idx+_delta_map]+_delta_out] = (decltype(_view_vec_out)::value_t)_view_vec_in[_idx];\n").c_str());
}

