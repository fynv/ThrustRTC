#include "scatter.h"

bool TRTC_Scatter(const DVVectorLike& vec_in, const DVVectorLike& vec_map, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_map", "view_vec_out", "begin_in", "begin_map", "begin_out" }, "idx",
		"    view_vec_out[view_vec_map[idx+begin_map]+begin_out] = (decltype(view_vec_out)::value_t)view_vec_in[idx + begin_in];\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_map(begin_map);
	DVSizeT dvbegin_out(begin_out);

	const DeviceViewable* args[] = { &vec_in, &vec_map, &vec_out, &dvbegin_in, &dvbegin_map, &dvbegin_out };
	return s_for.launch_n(end_in - begin_in, args);
}

bool TRTC_Scatter_If(const DVVectorLike& vec_in, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_stencil, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_map", "view_vec_stencil", "view_vec_out", "begin_in", "begin_map", "begin_stencil", "begin_out"}, "idx",
		"    if(view_vec_stencil[idx+begin_stencil])\n"
		"        view_vec_out[view_vec_map[idx+begin_map]+begin_out] = (decltype(view_vec_out)::value_t)view_vec_in[idx + begin_in];\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_map(begin_map);
	DVSizeT dvbegin_stencil(begin_stencil);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &vec_map, &vec_stencil, &vec_out, &dvbegin_in, &dvbegin_map, &dvbegin_stencil, &dvbegin_out };
	return s_for.launch_n(end_in - begin_in, args);
}

bool TRTC_Scatter_If(const DVVectorLike& vec_in, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_map, size_t begin_stencil, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_map", "view_vec_stencil", "view_vec_out", "pred", "begin_in", "begin_map", "begin_stencil", "begin_out" }, "idx",
		"    if(pred(view_vec_stencil[idx+begin_stencil]))\n"
		"        view_vec_out[view_vec_map[idx+begin_map]+begin_out] = (decltype(view_vec_out)::value_t)view_vec_in[idx + begin_in];\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_map(begin_map);
	DVSizeT dvbegin_stencil(begin_stencil);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &vec_map, &vec_stencil, &vec_out, &pred, &dvbegin_in, &dvbegin_map, &dvbegin_stencil, &dvbegin_out };
	return s_for.launch_n(end_in - begin_in, args);
}

