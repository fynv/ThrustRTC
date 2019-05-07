#include <memory.h>
#include "transform.h"

bool TRTC_Transform(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& op, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "delta_out", "op" }, "idx",
		"    view_vec_out[idx + delta_out] = op(view_vec_in[idx]);\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &dvdelta_out, &op};
	return s_for.launch(ctx, begin_in, end_in, args);
}

bool TRTC_Transform_Binary(TRTCContext& ctx, const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, DVVectorLike& vec_out, const Functor& op, size_t begin_in1, size_t end_in1, size_t begin_in2, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in1",  "view_vec_in2", "view_vec_out", "delta_in2", "delta_out", "op" }, "idx",
		"    view_vec_out[idx + delta_out] = op(view_vec_in1[idx], view_vec_in2[idx + delta_in2]);\n"
	);

	if (end_in1 == (size_t)(-1)) end_in1 = vec_in1.size();
	DVInt32 dvdelta_in2((int)begin_in2 - (int)begin_in1);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in1);
	const DeviceViewable* args[] = { &vec_in1, &vec_in2, &vec_out, &dvdelta_in2, &dvdelta_out, &op };
	return s_for.launch(ctx, begin_in1, end_in1, args);
}

bool TRTC_Transform_If(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& op, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "delta_out", "op", "pred" }, "idx",
		"    if (pred(view_vec_in[idx])) view_vec_out[idx + delta_out] = op(view_vec_in[idx]);\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &dvdelta_out, &op, &pred };
	return s_for.launch(ctx, begin_in, end_in, args);
}

bool TRTC_Transform_If_Stencil(TRTCContext& ctx, const DVVectorLike& vec_in, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& op, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_stencil", "view_vec_out", "delta_stencil", "delta_out", "op", "pred" }, "idx",
		"    if (pred(view_vec_stencil[idx +delta_stencil])) view_vec_out[idx + delta_out] = op(view_vec_in[idx]);\n"
	);
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta_stencil((int)begin_stencil - (int)begin_in);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_stencil, &vec_out, &dvdelta_stencil, &dvdelta_out, &op, &pred };
	return s_for.launch(ctx, begin_in, end_in, args);
}

bool TRTC_Transform_Binary_If_Stencil(TRTCContext& ctx, const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& op, const Functor& pred, size_t begin_in1, size_t end_in1, size_t begin_in2, size_t begin_stencil, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in1", "view_vec_in2", "view_vec_stencil", "view_vec_out", "delta_in2", "delta_stencil", "delta_out", "op", "pred" }, "idx",
		"    if (pred(view_vec_stencil[idx +delta_stencil])) view_vec_out[idx + delta_out] = op(view_vec_in1[idx], view_vec_in2[idx+delta_in2]);\n"
	);
	if (end_in1 == (size_t)(-1)) end_in1 = vec_in1.size();
	DVInt32 dvdelta_in2((int)begin_in2 - (int)begin_in1);
	DVInt32 dvdelta_stencil((int)begin_stencil - (int)begin_in1);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in1);
	const DeviceViewable* args[] = { &vec_in1, &vec_in2, &vec_stencil, &vec_out, &dvdelta_in2, &dvdelta_stencil, &dvdelta_out, &op, &pred };
	return s_for.launch(ctx, begin_in1, end_in1, args);
}
