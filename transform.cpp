#include "transform.h"

bool TRTC_Transform(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, const Functor& op, size_t begin_in, size_t end_in, size_t begin_out)
{
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);
	std::vector<TRTCContext::AssignedParam> arg_map = op.arg_map;
	arg_map.push_back({ "_view_vec_in", &vec_in });
	arg_map.push_back({ "_view_vec_out", &vec_out });
	arg_map.push_back({ "_delta_out", &dvdelta_out });

	if (end_in == (size_t)(-1)) end_in = vec_in.size();

	return ctx.launch_for(begin_in, end_in, arg_map, "_idx",
		(op.generate_code("decltype(_view_vec_out)::value_t", { "_view_vec_in[_idx]" }) +
			"     _view_vec_out[_idx+_delta_out] = " + op.functor_ret + "; \n").c_str());
}

bool TRTC_Transform_Binary(TRTCContext& ctx, const DVVector& vec_in1, const DVVector& vec_in2, DVVector& vec_out, const Functor& op, size_t begin_in1, size_t end_in1, size_t begin_in2, size_t begin_out)
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

	return ctx.launch_for(begin_in1, end_in1, arg_map, "_idx",
		(op.generate_code("decltype(_view_vec_out)::value_t", { "_view_vec_in1[_idx]", "_view_vec_in2[_idx +_delta_in2]" }) +
			"     _view_vec_out[_idx+_delta_out] = " + op.functor_ret + "; \n").c_str());
}

bool TRTC_Transform_If(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, const Functor& op, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_out)
{
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);
	std::vector<TRTCContext::AssignedParam> arg_map(op.arg_map.size() + pred.arg_map.size() + 3);
	memcpy(arg_map.data(), op.arg_map.data(), op.arg_map.size() * sizeof(TRTCContext::AssignedParam));
	memcpy(arg_map.data()+ op.arg_map.size(), pred.arg_map.data(), pred.arg_map.size() * sizeof(TRTCContext::AssignedParam));
	TRTCContext::AssignedParam* p_arg_map = &arg_map[op.arg_map.size() + pred.arg_map.size()];
	p_arg_map[0] = { "_view_vec_in", &vec_in };
	p_arg_map[1] = { "_view_vec_out", &vec_out };
	p_arg_map[2] = { "_delta_out", &dvdelta_out };

	if (end_in == (size_t)(-1)) end_in = vec_in.size();

	return ctx.launch_for(begin_in, end_in, arg_map, "_idx",
		(pred.generate_code("bool", {"_view_vec_in[_idx]"})+
			"    if (" + pred.functor_ret+")\n    {\n"+		
			op.generate_code("decltype(_view_vec_out)::value_t", { "_view_vec_in[_idx]" }) +
			"     _view_vec_out[_idx+_delta_out] = " + op.functor_ret + "; \n    }\n").c_str());
}

bool TRTC_Transform_If_Stencil(TRTCContext& ctx, const DVVector& vec_in, const DVVector& vec_stencil, DVVector& vec_out, const Functor& op, const Functor& pred, size_t begin_in, size_t end_in, size_t begin_stencil, size_t begin_out)
{
	DVInt32 dvdelta_stencil((int)begin_stencil - (int)begin_in);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in);
	std::vector<TRTCContext::AssignedParam> arg_map(op.arg_map.size() + pred.arg_map.size() + 5);
	memcpy(arg_map.data(), op.arg_map.data(), op.arg_map.size() * sizeof(TRTCContext::AssignedParam));
	memcpy(arg_map.data() + op.arg_map.size(), pred.arg_map.data(), pred.arg_map.size() * sizeof(TRTCContext::AssignedParam));
	TRTCContext::AssignedParam* p_arg_map = &arg_map[op.arg_map.size() + pred.arg_map.size()];
	p_arg_map[0] = { "_view_vec_in", &vec_in };
	p_arg_map[1] = { "_view_vec_stencil", &vec_stencil };
	p_arg_map[2] = { "_view_vec_out", &vec_out };
	p_arg_map[3] = { "_delta_stencil", &dvdelta_stencil };
	p_arg_map[4] = { "_delta_out", &dvdelta_out };

	if (end_in == (size_t)(-1)) end_in = vec_in.size();

	return ctx.launch_for(begin_in, end_in, arg_map, "_idx",
		(pred.generate_code("bool", { "_view_vec_stencil[_idx +_delta_stencil]" }) +
			"    if (" + pred.functor_ret + ")\n    {\n" +
			op.generate_code("decltype(_view_vec_out)::value_t", { "_view_vec_in[_idx]" }) +
			"     _view_vec_out[_idx+_delta_out] = " + op.functor_ret + "; \n    }\n").c_str());
}

bool THRUST_RTC_API TRTC_Transform_Binary_If_Stencil(TRTCContext& ctx, const DVVector& vec_in1, const DVVector& vec_in2, const DVVector& vec_stencil, DVVector& vec_out, const Functor& op, const Functor& pred, size_t begin_in1, size_t end_in1, size_t begin_in2, size_t begin_stencil, size_t begin_out)
{
	DVInt32 dvdelta_in2((int)begin_in2 - (int)begin_in1);
	DVInt32 dvdelta_stencil((int)begin_stencil - (int)begin_in1);
	DVInt32 dvdelta_out((int)begin_out - (int)begin_in1);
	std::vector<TRTCContext::AssignedParam> arg_map(op.arg_map.size() + pred.arg_map.size() + 7);
	memcpy(arg_map.data(), op.arg_map.data(), op.arg_map.size() * sizeof(TRTCContext::AssignedParam));
	memcpy(arg_map.data() + op.arg_map.size(), pred.arg_map.data(), pred.arg_map.size() * sizeof(TRTCContext::AssignedParam));
	TRTCContext::AssignedParam* p_arg_map = &arg_map[op.arg_map.size() + pred.arg_map.size()];
	p_arg_map[0] = { "_view_vec_in1", &vec_in1 };
	p_arg_map[1] = { "_view_vec_in2", &vec_in2 };
	p_arg_map[2] = { "_view_vec_stencil", &vec_stencil };
	p_arg_map[3] = { "_view_vec_out", &vec_out };
	p_arg_map[4] = { "_delta_in2", &dvdelta_in2 };
	p_arg_map[5] = { "_delta_stencil", &dvdelta_stencil };
	p_arg_map[6] = { "_delta_out", &dvdelta_out };

	if (end_in1 == (size_t)(-1)) end_in1 = vec_in1.size();

	return ctx.launch_for(begin_in1, end_in1, arg_map, "_idx",
		(pred.generate_code("bool", { "_view_vec_stencil[_idx +_delta_stencil]" }) +
			"    if (" + pred.functor_ret + ")\n    {\n" +
			op.generate_code("decltype(_view_vec_out)::value_t", { "_view_vec_in1[_idx]", "_view_vec_in2[_idx+_delta_in2]" }) +
			"     _view_vec_out[_idx+_delta_out] = " + op.functor_ret + "; \n    }\n").c_str());
}
