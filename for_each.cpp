#include "for_each.h"
#include "DeviceViewable.h"

bool TRTC_For_Each(TRTCContext& ctx, DVVectorLike& vec, const Functor& f, size_t begin, size_t end)
{
	std::vector<TRTCContext::AssignedParam> arg_map = f.arg_map;
	arg_map.push_back({ "_view_vec", &vec });

	if (end == (size_t)(-1)) end = vec.size();

	return ctx.launch_for(begin, end, arg_map, "_idx", f.generate_code(nullptr, {"_view_vec[_idx]"}).c_str());
}
