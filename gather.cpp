#include "gather.h"

bool TRTC_Gather(TRTCContext& ctx, const DVVectorLike& vec_map, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_map, size_t end_map, size_t begin_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_map", "view_vec_in", "view_vec_out", "delta_in", "delta_out" }, "idx",
		"    view_vec_out[idx+delta_out] = (decltype(view_vec_out)::value_t)view_vec_in[view_vec_map[idx]+ delta_in];\n"
	);

	DVInt32 dvdelta_in((int)begin_in - (int)begin_map);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_map);

	if (end_map == (size_t)(-1)) end_map = vec_map.size();
	const DeviceViewable* args[] = { &vec_map, &vec_in, &vec_out, &dvdelta_in, &dvdelta_out };
	return s_for.launch(ctx, begin_map, end_map, args);	
}

bool TRTC_Gather_If(TRTCContext& ctx, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_map, size_t end_map, size_t begin_stencil, size_t begin_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_map", "view_vec_stencil", "view_vec_in", "view_vec_out", "delta_stencil", "delta_in", "delta_out" }, "idx",
		"    if(view_vec_stencil[idx+delta_stencil])\n"
		"        view_vec_out[idx+delta_out] = (decltype(view_vec_out)::value_t)view_vec_in[view_vec_map[idx]+ delta_in];\n"
	);

	DVInt32 dvdelta_stencil((int)begin_stencil - (int)begin_map);
	DVInt32 dvdelta_in((int)begin_in - (int)begin_map);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_map);

	if (end_map == (size_t)(-1)) end_map = vec_map.size();
	const DeviceViewable* args[] = { &vec_map, &vec_stencil, &vec_in, &vec_out, &dvdelta_stencil, &dvdelta_in, &dvdelta_out };
	return s_for.launch(ctx, begin_map, end_map, args);
}

bool TRTC_Gather_If(TRTCContext& ctx, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred, size_t begin_map, size_t end_map, size_t begin_stencil, size_t begin_in, size_t begin_out)
{
	DVInt32 dvdelta_stencil((int)begin_stencil - (int)begin_map);
	DVInt32 dvdelta_in((int)begin_in - (int)begin_map);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_map);
	std::vector<TRTCContext::AssignedParam> arg_map = pred.arg_map;
	arg_map.push_back({ "_view_vec_map", &vec_map });
	arg_map.push_back({ "_view_vec_stencil", &vec_stencil });
	arg_map.push_back({ "_view_vec_in", &vec_in });
	arg_map.push_back({ "_view_vec_out", &vec_out });
	arg_map.push_back({ "_delta_stencil", &dvdelta_stencil });
	arg_map.push_back({ "_delta_in", &dvdelta_in });
	arg_map.push_back({ "_delta_out", &dvdelta_out });

	if (end_map == (size_t)(-1)) end_map = vec_map.size();

	return ctx.launch_for(begin_map, end_map, arg_map, "_idx",
		(pred.generate_code("bool", { "_view_vec_stencil[_idx+_delta_stencil]" }) +
			"    if("+ pred.functor_ret + ")\n"
			"        _view_vec_out[_idx+_delta_out] = (decltype(_view_vec_out)::value_t)_view_vec_in[_view_vec_map[_idx]+ _delta_in];\n").c_str());
}
