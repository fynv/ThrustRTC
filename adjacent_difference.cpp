#include "adjacent_difference.h"

bool TRTC_Adjacent_Difference(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for({ "view_vec_in", "view_vec_out", "delta"}, "idx",
	"    auto value = view_vec_in[idx];\n"
	"    if (idx>0) value -= view_vec_in[idx-1]; \n"
	"    view_vec_out[idx+delta] = (decltype(view_vec_out)::value_t) value;\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &dvdelta };
	s_for.launch(ctx, begin_in, end_in, args);
	return true;
}

bool TRTC_Adjacent_Difference(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, const Functor& binary_op, size_t begin_in, size_t end_in, size_t begin_out)
{
	std::vector<TRTCContext::AssignedParam> arg_map = binary_op.arg_map;
	arg_map.push_back({ "_view_vec_in", &vec_in });
	arg_map.push_back({ "_view_vec_out", &vec_out });
	DVInt32 dvdelta((int)begin_out - (int)begin_in);
	arg_map.push_back({ "_delta", &dvdelta });

	if (end_in == (size_t)(-1)) end_in = vec_in.size();

	ctx.launch_for(begin_in, end_in, arg_map, "_idx",
		(std::string("    auto value = _view_vec_in[_idx];\n") +
		"    if (_idx>0)\n    {\n" +
		binary_op.generate_code("decltype(_view_vec_in)::value_t", { "_view_vec_in[_idx]", "_view_vec_in[_idx - 1]" }) +
		"    value = " + binary_op.functor_ret + ";\n"
		"    }\n"
		"    _view_vec_out[_idx+_delta] = (decltype(_view_vec_out)::value_t) value;\n").c_str());

	return true;
}
