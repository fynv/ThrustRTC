#include "logical.h"

bool TRTC_All_Of(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, bool& ret, size_t begin, size_t end)
{
	ret = true;
	DVVector dvres(ctx, "bool", 1, &ret);
	std::vector<TRTCContext::AssignedParam> arg_map = pred.arg_map;
	arg_map.push_back({ "_view_vec", &vec });
	arg_map.push_back({ "_view_res", &dvres });

	if (end == (size_t)(-1)) end = vec.size();
	ret = false;
	if (end - begin < 1) return true;
	if (!ctx.launch_for(begin, end, arg_map, "_idx",
		(pred.generate_code("bool", { "_view_vec[_idx]" }) +
			"     if(!" + pred.functor_ret + ")  _view_res[0]=false;\n").c_str())) return false;
	dvres.to_host(&ret);
	return true;
}

bool TRTC_Any_Of(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, bool& ret, size_t begin, size_t end)
{
	ret = false;
	DVVector dvres(ctx, "bool", 1, &ret);
	std::vector<TRTCContext::AssignedParam> arg_map = pred.arg_map;
	arg_map.push_back({ "_view_vec", &vec });
	arg_map.push_back({ "_view_res", &dvres });

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;
	if (!ctx.launch_for(begin, end, arg_map, "_idx",
		(pred.generate_code("bool", { "_view_vec[_idx]" }) +
			"     if(" + pred.functor_ret + ")  _view_res[0]=true;\n").c_str())) return false;
	dvres.to_host(&ret);
	return true;
}


bool TRTC_None_Of(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, bool& ret, size_t begin, size_t end)
{
	ret = true;
	DVVector dvres(ctx, "bool", 1, &ret);
	std::vector<TRTCContext::AssignedParam> arg_map = pred.arg_map;
	arg_map.push_back({ "_view_vec", &vec });
	arg_map.push_back({ "_view_res", &dvres });

	if (end == (size_t)(-1)) end = vec.size();
	if (end - begin < 1) return true;
	if (!ctx.launch_for(begin, end, arg_map, "_idx",
		(pred.generate_code("bool", { "_view_vec[_idx]" }) +
			"     if(" + pred.functor_ret + ")  _view_res[0]=false;\n").c_str())) return false;
	dvres.to_host(&ret);
	return true;
}

