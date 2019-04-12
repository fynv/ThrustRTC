#include "for_each.h"
#include "for.h"
#include "DeviceViewable.h"

void TRTC_For_Each(TRTCContext& ctx, DVVector& vec, const Functor& f, size_t begin, size_t end)
{
	std::vector<TRTCContext::AssignedParam> arg_map = f.arg_map;
	arg_map.push_back({ "_view_vec", &vec });

	if (end == (size_t)(-1)) end = vec.size();

	TRTC_For_Once(ctx, begin, end, arg_map, "_idx",
		(std::string("    do{\n")+
		"        " + vec.name_elem_cls() + " " + f.functor_params[0] + " = _view_vec[_idx];\n" +
		f.code_body +
		"    } while(false);\n").c_str());

}
