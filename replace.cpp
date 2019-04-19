#include "replace.h"

bool TRTC_Replace(TRTCContext& ctx, DVVector& vec, const DeviceViewable& old_value, const DeviceViewable& new_value, size_t begin, size_t end)
{
	static TRTC_For s_for( {"view_vec", "old_value", "new_value" }, "idx",
		"    if (view_vec[idx]==(decltype(view_vec)::value_t)old_value) view_vec[idx] = (decltype(view_vec)::value_t)new_value;\n"
	);

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &old_value, &new_value };
	return s_for.launch(ctx, begin, end, args);
}

bool TRTC_Replace_If(TRTCContext& ctx, DVVector& vec, const Functor& pred, const DeviceViewable& new_value, size_t begin, size_t end)
{
	std::vector<TRTCContext::AssignedParam> arg_map = pred.arg_map;
	arg_map.push_back({ "_view_vec", &vec });
	arg_map.push_back({ "_new_value", &new_value });

	if (end == (size_t)(-1)) end = vec.size();

	return ctx.launch_for(begin, end, arg_map, "_idx",
		(pred.generate_code("bool", {"_view_vec[_idx]"})+
		"    if (" + pred.functor_ret + ") _view_vec[_idx] = (decltype(_view_vec)::value_t)_new_value; \n").c_str());
}

bool TRTC_Replace_Copy(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, const DeviceViewable& old_value, const DeviceViewable& new_value, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For s_for(
	{ "view_vec_in", "view_vec_out" , "old_value", "new_value", "delta" }, "idx",
	"    auto value = view_vec_in[idx];\n"
	"    view_vec_out[idx+delta] = value == (decltype(view_vec_in)::value_t)old_value ?  (decltype(view_vec_out)::value_t)new_value :  (decltype(view_vec_out)::value_t)value;\n"
	);

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &old_value, &new_value, &dvdelta };
	return s_for.launch(ctx, begin_in, end_in, args);
}

bool TRTC_Replace_Copy_If(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, const Functor& pred, const DeviceViewable& new_value, size_t begin_in, size_t end_in, size_t begin_out)
{
	std::vector<TRTCContext::AssignedParam> arg_map = pred.arg_map;
	arg_map.push_back({ "_view_vec_in", &vec_in });
	arg_map.push_back({ "_view_vec_out", &vec_out });
	arg_map.push_back({ "_new_value", &new_value });
	DVInt32 dvdelta((int)begin_out - (int)begin_in);
	arg_map.push_back({ "_delta", &dvdelta });
	
	if (end_in == (size_t)(-1)) end_in = vec_in.size();

	return ctx.launch_for( begin_in, end_in, arg_map, "_idx",
		(pred.generate_code("bool", { "_view_vec_in[_idx]" }) +
		"    _view_vec_out[_idx+_delta] = " + pred.functor_ret + "?  (decltype(_view_vec_out)::value_t)_new_value : (decltype(_view_vec_out)::value_t)_view_vec_in[_idx]; \n").c_str());
}