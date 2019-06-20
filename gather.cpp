#include "gather.h"

bool TRTC_Gather(const DVVectorLike& vec_map, const DVVectorLike& vec_in, DVVectorLike& vec_out)
{
	static TRTC_For s_for({ "view_vec_map", "view_vec_in", "view_vec_out" }, "idx",
		"    view_vec_out[idx] = (decltype(view_vec_out)::value_t)view_vec_in[view_vec_map[idx]];\n"
	);

	const DeviceViewable* args[] = { &vec_map, &vec_in, &vec_out };
	return s_for.launch_n(vec_map.size(), args);
}

bool TRTC_Gather_If(const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, const DVVectorLike& vec_in, DVVectorLike& vec_out)
{
	static TRTC_For s_for({ "view_vec_map", "view_vec_stencil", "view_vec_in", "view_vec_out" }, "idx",
		"    if(view_vec_stencil[idx])\n"
		"        view_vec_out[idx] = (decltype(view_vec_out)::value_t)view_vec_in[view_vec_map[idx]];\n"
	);

	const DeviceViewable* args[] = { &vec_map, &vec_stencil, &vec_in, &vec_out };
	return s_for.launch_n(vec_map.size(), args);
}

bool TRTC_Gather_If(const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred)
{
	static TRTC_For s_for({ "view_vec_map", "view_vec_stencil", "view_vec_in", "view_vec_out", "pred"}, "idx",
		"    if(pred(view_vec_stencil[idx]))\n"
		"        view_vec_out[idx] = (decltype(view_vec_out)::value_t)view_vec_in[view_vec_map[idx]];\n"
	);

	const DeviceViewable* args[] = { &vec_map, &vec_stencil, &vec_in, &vec_out, &pred };
	return s_for.launch_n(vec_map.size(), args);
}
