#include "replace.h"
#include "for.h"

bool TRTC_Replace(TRTCContext& ctx, DVVector& vec, const DeviceViewable& old_value, const DeviceViewable& new_value, size_t begin, size_t end)
{
	static TRTC_For_Template s_templ(
	{ "T" },
	{ { "VectorView<T>", "view_vec" }, { "T", "old_value" },{ "T", "new_value" } }, "idx",
		"    if (view_vec[idx]==old_value) view_vec[idx] = new_value;\n"
	);

	if (vec.name_elem_cls() != old_value.name_view_cls())
	{
		printf("TRTC_Replace: vector type mismatch with old_value type.\n");
		return false;
	}

	if (vec.name_elem_cls() != new_value.name_view_cls())
	{
		printf("TRTC_Replace: vector type mismatch with new_value type.\n");
		return false;
	}

	if (end == (size_t)(-1)) end = vec.size();
	const DeviceViewable* args[] = { &vec, &old_value, &new_value };
	s_templ.launch(ctx, begin, end, args);
	return true;

}

bool TRTC_Replace_If(TRTCContext& ctx, DVVector& vec, const Functor& pred, const DeviceViewable& new_value, size_t begin, size_t end)
{
	if (vec.name_elem_cls() != new_value.name_view_cls())
	{
		printf("TRTC_Replace_If: vector type mismatch with new_value type.\n");
		return false;
	}

	std::vector<TRTCContext::AssignedParam> arg_map = pred.arg_map;
	arg_map.push_back({ "_view_vec", &vec });
	arg_map.push_back({ "_new_value", &new_value });

	if (end == (size_t)(-1)) end = vec.size();

	TRTC_For_Once(ctx, begin, end, arg_map, "_idx",
		(std::string("    bool ") + pred.functor_ret + "=false;\n"
		"    do{\n"
		"        " + vec.name_elem_cls() + " " + pred.functor_params[0] + " = _view_vec[_idx];\n" +
		pred.code_body + 
		"    } while(false);\n"
		"    if (" + pred.functor_ret + ") _view_vec[_idx] = _new_value; \n").c_str());

	return true;
}

bool TRTC_Replace_Copy(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, const DeviceViewable& old_value, const DeviceViewable& new_value, size_t begin_in, size_t end_in, size_t begin_out)
{
	static TRTC_For_Template s_templ(
	{ "T" },
	{ { "VectorView<T>", "view_vec_in" }, { "VectorView<T>", "view_vec_out" }, { "T", "old_value" }, { "T", "new_value" }, { "int32_t", "delta" } }, "idx",
	"    T value = view_vec_in[idx];\n"
	"    view_vec_out[idx+delta] = value == old_value ? new_value : value;\n"
	);

	if (vec_in.name_elem_cls() != vec_out.name_elem_cls())
	{
		printf("TRTC_Replace_Copy: input vector type mismatch with output vector type.\n");
		return false;
	}

	if (vec_in.name_elem_cls() != old_value.name_view_cls())
	{
		printf("TRTC_Replace_Copy: vector type mismatch with old_value type.\n");
		return false;
	}

	if (vec_in.name_elem_cls() != new_value.name_view_cls())
	{
		printf("TRTC_Replace_Copy: vector type mismatch with new_value type.\n");
		return false;
	}

	if (end_in == (size_t)(-1)) end_in = vec_in.size();
	DVInt32 dvdelta((int)begin_out - (int)begin_in);
	const DeviceViewable* args[] = { &vec_in, &vec_out, &old_value, &new_value, &dvdelta };
	s_templ.launch(ctx, begin_in, end_in, args);
	return true;
}

bool TRTC_Replace_Copy_If(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, const Functor& pred, const DeviceViewable& new_value, size_t begin_in, size_t end_in, size_t begin_out)
{
	if (vec_in.name_elem_cls() != vec_out.name_elem_cls())
	{
		printf("TRTC_Replace_Copy_If: input vector type mismatch with output vector type.\n");
		return false;
	}

	if (vec_in.name_elem_cls() != new_value.name_view_cls())
	{
		printf("TRTC_Replace_Copy_If: vector type mismatch with new_value type.\n");
		return false;
	}

	std::vector<TRTCContext::AssignedParam> arg_map = pred.arg_map;
	arg_map.push_back({ "_view_vec_in", &vec_in });
	arg_map.push_back({ "_view_vec_out", &vec_out });
	arg_map.push_back({ "_new_value", &new_value });
	DVInt32 dvdelta((int)begin_out - (int)begin_in);
	arg_map.push_back({ "_delta", &dvdelta });
	
	if (end_in == (size_t)(-1)) end_in = vec_in.size();

	TRTC_For_Once(ctx, begin_in, end_in, arg_map, "_idx",
		(std::string("    bool ") + pred.functor_ret + "=false;\n"
		"    do{\n"
		"        " + vec_in.name_elem_cls() + " " + pred.functor_params[0] + " = _view_vec_in[_idx];\n" +
		pred.code_body +
		"    } while(false);\n"
		"    _view_vec_out[_idx+_delta] = " + pred.functor_ret + "? _new_value : _view_vec_in[_idx]; \n").c_str());

	return true;
}