#include "transform.h"

void TRTC_transform(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, const Functor& op, size_t begin_in, size_t end_in, size_t begin_out)
{
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);
	std::vector<TRTCContext::AssignedParam> arg_map = op.arg_map;
	arg_map.push_back({ "_view_vec_in", &vec_in });
	arg_map.push_back({ "_view_vec_out", &vec_out });
	arg_map.push_back({ "_delta_out", &dvdelta_out });

	if (end_in == (size_t)(-1)) end_in = vec_in.size();

	ctx.launch_for(begin_in, end_in, arg_map, "_idx",
		(op.generate_code("decltype(_view_vec_out)::value_t", { "_view_vec_in[_idx]" }) +
			"     _view_vec_out[_idx+_delta_out] = " + op.functor_ret + "; \n").c_str());
}

void TRTC_transform(TRTCContext& ctx, const DVVector& vec_in1, const DVVector& vec_in2, DVVector& vec_out, const Functor& op, size_t begin_in1, size_t end_in1, size_t begin_in2, size_t begin_out)
{
	DVInt32 dvdelta_in2((int)begin_in2 - (int)begin_in1);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in1);
	std::vector<TRTCContext::AssignedParam> arg_map = op.arg_map;
	arg_map.push_back({ "_view_vec_in1", &vec_in1 });
	arg_map.push_back({ "_view_vec_in2", &vec_in2 });
	arg_map.push_back({ "_view_vec_out", &vec_out });
	arg_map.push_back({ "_delta_in2", &dvdelta_in2 });
	arg_map.push_back({ "_delta_out", &dvdelta_out });

	if (end_in1 == (size_t)(-1)) end_in1 = vec_in1.size();

	ctx.launch_for(begin_in1, end_in1, arg_map, "_idx",
		(op.generate_code("decltype(_view_vec_out)::value_t", { "_view_vec_in1[_idx]", "_view_vec_in2[_idx+_delta_in2]" }) +
			"     _view_vec_out[_idx+_delta_out] = " + op.functor_ret + "; \n").c_str());
}

