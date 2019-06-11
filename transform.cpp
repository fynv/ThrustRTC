#include <memory.h>
#include "transform.h"

bool TRTC_Transform(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& op, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "begin_in", "begin_out" , "op" }, "idx",
		"    view_vec_out[idx + begin_out] = op(view_vec_in[idx + begin_in]);\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &dvbegin_in, &dvbegin_out, &op};
	return s_for.launch_n(end_in - begin_in, args);
}

bool TRTC_Transform_Binary(const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, DVVectorLike& vec_out, const Functor& op, size_t begin_in1, size_t end_in1, size_t begin_in2, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in1",  "view_vec_in2", "view_vec_out", "begin_in1", "begin_in2", "begin_out", "op" }, "idx",
		"    view_vec_out[idx + begin_out] = op(view_vec_in1[idx + begin_in1], view_vec_in2[idx + begin_in2]);\n"
	);

	if (end_in1 == (size_t)(-1)) end_in1 = vec_in1.size();
	DVSizeT dvbegin_in1(begin_in1);
	DVSizeT dvbegin_in2(begin_in2);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in1, &vec_in2, &vec_out, &dvbegin_in1, &dvbegin_in2, &dvbegin_out, &op };
	return s_for.launch_n(end_in1 - begin_in1, args);
}

bool TRTC_Transform_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& op, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "begin_in", "begin_out", "op", "pred" }, "idx",
		"    if (pred(view_vec_in[idx + begin_in])) view_vec_out[idx + begin_in] = op(view_vec_in[idx + begin_in]);\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &dvbegin_in, &dvbegin_out, &op, &pred };
	return s_for.launch_n(end_in - begin_in, args);
}

bool TRTC_Transform_If_Stencil(const DVVectorLike& vec_in, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& op, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_stencil", "view_vec_out", "begin_in", "begin_stencil", "begin_out", "op", "pred" }, "idx",
		"    if (pred(view_vec_stencil[idx + begin_stencil])) view_vec_out[idx + begin_out] = op(view_vec_in[idx + begin_in]);\n"
	);
	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_stencil(begin_stencil);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &vec_stencil, &vec_out, &dvbegin_in, &dvbegin_stencil, &dvbegin_out, &op, &pred };
	return s_for.launch_n(end_in - begin_in, args);
}

bool TRTC_Transform_Binary_If_Stencil(const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& op, const Functor& pred, size_t begin_in1, size_t end_in1, size_t begin_in2, size_t begin_stencil, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in1", "view_vec_in2", "view_vec_stencil", "view_vec_out", "begin_in1", "begin_in2", "begin_stencil", "begin_out", "op", "pred" }, "idx",
		"    if (pred(view_vec_stencil[idx + begin_stencil])) view_vec_out[idx + begin_out] = op(view_vec_in1[idx + begin_in1], view_vec_in2[idx+begin_in2]);\n"
	);
	if (end_in1 == (size_t)(-1)) end_in1 = vec_in1.size();
	DVSizeT dvbegin_in1(begin_in1);
	DVSizeT dvbegin_in2(begin_in2);
	DVSizeT dvbegin_stencil(begin_stencil);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in1, &vec_in2, &vec_stencil, &vec_out, &dvbegin_in1, &dvbegin_in2, &dvbegin_stencil, &dvbegin_out, &op, &pred };
	return s_for.launch_n(end_in1 - begin_in1, args);
}
