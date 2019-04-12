#include "adjacent_difference.h"
#include "for.h"

bool TRTC_Adjacent_Difference(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For_Template s_templ(
	{ "T" },
	{ { "VectorView<T>", "view_vec_in" }, { "VectorView<T>", "view_vec_out" }, { "int32_t", "delta" } }, "idx",
	"    T value = view_vec_in[idx];\n"
	"    if (idx>0) value -= view_vec_in[idx-1]; \n"
	"    view_vec_out[idx+delta] = value;\n"
	);

	if (vec_in.name_elem_cls() != vec_out.name_elem_cls())
	{
		printf("TRTC_Adjacent_Difference: input vector type mismatch with output vector type.\n");
		return false;
	}

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &dvdelta };
	s_templ.launch(ctx, begin_in, end_in, args);
	return true;
}

bool TRTC_Adjacent_Difference(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, const Functor& binary_op, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (vec_in.name_elem_cls() != vec_out.name_elem_cls())
	{
		printf("TRTC_Adjacent_Difference: input vector type mismatch with output vector type.\n");
		return false;
	}

	std::vector<TRTCContext::AssignedParam> arg_map = binary_op.arg_map;
	arg_map.push_back({ "_view_vec_in", &vec_in });
	arg_map.push_back({ "_view_vec_out", &vec_out });
	DVInt32 dvdelta((int)begin_out - (int)begin_in);
	arg_map.push_back({ "_delta", &dvdelta });

	if (end_in == (size_t)(-1)) end_in = vec_in.size();

	TRTC_For_Once(ctx, begin_in, end_in, arg_map, "_idx",
		(std::string("    ") + vec_in.name_elem_cls() + " " + binary_op.functor_ret + " = _view_vec_in[_idx];\n"
		"    if (_idx>0)\n    {\n"
		"        do{\n"
		"            " + vec_in.name_elem_cls() + " " + binary_op.functor_params[0] + " = _view_vec_in[_idx];\n"
		"            " + vec_in.name_elem_cls() + " " + binary_op.functor_params[1] + " = _view_vec_in[_idx - 1];\n" +
		binary_op.code_body +
		"        } while(false);\n"
		"    }\n"
		"    _view_vec_out[_idx+_delta] = " + binary_op.functor_ret + ";\n").c_str());
	return true;
}
