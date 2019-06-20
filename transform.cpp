#include <memory.h>
#include "transform.h"

bool TRTC_Transform(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& op)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "op" }, "idx",
		"    view_vec_out[idx] = op(view_vec_in[idx]);\n"
	);

	const DeviceViewable* args[] = { &vec_in, &vec_out, &op};
	return s_for.launch_n(vec_in.size(), args);
}

bool TRTC_Transform_Binary(const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, DVVectorLike& vec_out, const Functor& op)
{
	static TRTC_For s_for({ "view_vec_in1",  "view_vec_in2", "view_vec_out", "op" }, "idx",
		"    view_vec_out[idx] = op(view_vec_in1[idx], view_vec_in2[idx]);\n"
	);

	const DeviceViewable* args[] = { &vec_in1, &vec_in2, &vec_out, &op };
	return s_for.launch_n(vec_in1.size(), args);
}

bool TRTC_Transform_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& op, const Functor& pred)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "op", "pred" }, "idx",
		"    if (pred(view_vec_in[idx])) view_vec_out[idx] = op(view_vec_in[idx]);\n"
	);

	const DeviceViewable* args[] = { &vec_in, &vec_out, &op, &pred };
	return s_for.launch_n(vec_in.size(), args);
}

bool TRTC_Transform_If_Stencil(const DVVectorLike& vec_in, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& op, const Functor& pred)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_stencil", "view_vec_out", "op", "pred" }, "idx",
		"    if (pred(view_vec_stencil[idx])) view_vec_out[idx] = op(view_vec_in[idx]);\n"
	);
	const DeviceViewable* args[] = { &vec_in, &vec_stencil, &vec_out, &op, &pred };
	return s_for.launch_n(vec_in.size(), args);
}

bool TRTC_Transform_Binary_If_Stencil(const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& op, const Functor& pred)
{
	static TRTC_For s_for({ "view_vec_in1", "view_vec_in2", "view_vec_stencil", "view_vec_out", "op", "pred" }, "idx",
		"    if (pred(view_vec_stencil[idx ])) view_vec_out[idx] = op(view_vec_in1[idx], view_vec_in2[idx]);\n"
	);
	
	const DeviceViewable* args[] = { &vec_in1, &vec_in2, &vec_stencil, &vec_out, &op, &pred };
	return s_for.launch_n(vec_in1.size(), args);
}
