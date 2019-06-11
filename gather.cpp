#include "gather.h"

bool TRTC_Gather(const DVVectorLike& vec_map, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_map, size_t end_map, size_t begin_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_map", "view_vec_in", "view_vec_out", "begin_map", "begin_in", "begin_out" }, "idx",
		"    view_vec_out[idx+begin_out] = (decltype(view_vec_out)::value_t)view_vec_in[view_vec_map[idx+begin_map]+ begin_in];\n"
	);

	if (end_map == (size_t)(-1)) end_map = vec_map.size();
	DVSizeT dvbegin_map(begin_map);
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);

	const DeviceViewable* args[] = { &vec_map, &vec_in, &vec_out, &dvbegin_map, &dvbegin_in, &dvbegin_out };
	return s_for.launch_n(end_map - begin_map, args);
}

bool TRTC_Gather_If(const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_map, size_t end_map, size_t begin_stencil, size_t begin_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_map", "view_vec_stencil", "view_vec_in", "view_vec_out", "begin_map", "begin_stencil", "begin_in", "begin_out" }, "idx",
		"    if(view_vec_stencil[idx+begin_stencil])\n"
		"        view_vec_out[idx+begin_out] = (decltype(view_vec_out)::value_t)view_vec_in[view_vec_map[idx +begin_map]+ begin_in];\n"
	);

	if (end_map == (size_t)(-1)) end_map = vec_map.size();
	DVSizeT dvbegin_map(begin_map);
	DVSizeT dvbegin_stencil(begin_stencil);
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_map, &vec_stencil, &vec_in, &vec_out, &dvbegin_map, &dvbegin_stencil, &dvbegin_in, &dvbegin_out };
	return s_for.launch_n(end_map - begin_map, args);
}

bool TRTC_Gather_If(const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred, size_t begin_map, size_t end_map, size_t begin_stencil, size_t begin_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_map", "view_vec_stencil", "view_vec_in", "view_vec_out", "pred", "begin_map", "begin_stencil", "begin_in", "begin_out" }, "idx",
		"    if(pred(view_vec_stencil[idx+begin_stencil]))\n"
		"        view_vec_out[idx+begin_out] = (decltype(view_vec_out)::value_t)view_vec_in[view_vec_map[idx +begin_map]+ begin_in];\n"
	);

	if (end_map == (size_t)(-1)) end_map = vec_map.size();
	DVSizeT dvbegin_map(begin_map);
	DVSizeT dvbegin_stencil(begin_stencil);
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_map, &vec_stencil, &vec_in, &vec_out, &pred, &dvbegin_map, &dvbegin_stencil, &dvbegin_in, &dvbegin_out };
	return s_for.launch_n(end_map - begin_map, args);
}
