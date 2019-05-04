#include "equal.h"

bool TRTC_Equal(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, bool& ret, size_t begin1, size_t end1, size_t begin2)
{
	ret = true;
	DVVector dvres(ctx, "bool", 1, &ret);
	DVInt32 dvdelta2((int)begin2 - (int)begin1);
	static TRTC_For s_for({ "view_vec1", "view_vec2", "view_res", "delta2" }, "idx",
		"    if (view_vec1[idx]!=(decltype(view_vec1)::value_t)view_vec2[idx + delta2]) view_res[0]=false;\n"
	);

	if (end1 == (size_t)(-1)) end1 = vec1.size();
	const DeviceViewable* args[] = { &vec1, &vec2, &dvres, &dvdelta2 };
	if (!s_for.launch(ctx, begin1, end1, args)) return false;
	dvres.to_host(&ret);
	return true;
}

bool TRTC_Equal(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, const Functor& binary_pred, bool& ret, size_t begin1, size_t end1, size_t begin2)
{
	ret = true;
	DVVector dvres(ctx, "bool", 1, &ret);
	DVInt32 dvdelta2((int)begin2 - (int)begin1);
	std::vector<TRTCContext::AssignedParam> arg_map = binary_pred.arg_map;
	arg_map.push_back({ "_view_vec1", &vec1 });
	arg_map.push_back({ "_view_vec2", &vec2 });
	arg_map.push_back({ "_view_res", &dvres });
	arg_map.push_back({ "_delta2", &dvdelta2 });

	if (end1 == (size_t)(-1)) end1 = vec1.size();
	if (!ctx.launch_for(begin1, end1, arg_map, "_idx",
		(binary_pred.generate_code("bool", { "_view_vec1[_idx]", "_view_vec2[_idx + _delta2]" }) +
			"     if(!" + binary_pred.functor_ret + ")  _view_res[0]=false;\n").c_str())) return false;
	dvres.to_host(&ret);
	return true;
}