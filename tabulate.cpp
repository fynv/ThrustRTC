#include "tabulate.h"

void TRTC_tabulate(TRTCContext& ctx, DVVector& vec, const Functor& op, size_t begin, size_t end)
{
	std::vector<TRTCContext::AssignedParam> arg_map = op.arg_map;
	arg_map.push_back({ "_view_vec", &vec });

	if (end == (size_t)(-1)) end = vec.size();

	ctx.launch_for(begin, end, arg_map, "_idx",
		(op.generate_code("decltype(_view_vec)::value_t", { "_view_vec[_idx]" }) +
		"     _view_vec[_idx] = " + op.functor_ret + "; \n").c_str());
}
