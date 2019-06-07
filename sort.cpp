#include "sort.h"
#include "merge_sort.h"
#include "radix_sort.h"

#define BLOCK_SIZE 256

bool TRTC_Sort(TRTCContext& ctx, DVVectorLike& vec, const Functor& comp, size_t begin, size_t end)
{
	if (comp.name_view_cls() == "Less")
	{
		std::string elem_type = vec.name_elem_cls();

		if (elem_type == "int8_t" || elem_type == "uint8_t" || elem_type == "int16_t" || elem_type == "uint16_t"
			|| elem_type == "int32_t" || elem_type == "uint32_t" || elem_type == "float")
			return radix_sort_32(ctx, vec, begin, end);
		else if  (elem_type == "int64_t" || elem_type == "uint64_t" || elem_type == "double")
			return radix_sort_64(ctx, vec, begin, end);
	}
	else if (comp.name_view_cls() == "Greater")
	{
		std::string elem_type = vec.name_elem_cls();
		if (elem_type == "int8_t" || elem_type == "uint8_t" || elem_type == "int16_t" || elem_type == "uint16_t"
			|| elem_type == "int32_t" || elem_type == "uint32_t" || elem_type == "float")
			return radix_sort_reverse_32(ctx, vec, begin, end);
		else if (elem_type == "int64_t" || elem_type == "uint64_t" || elem_type == "double")
			return radix_sort_reverse_64(ctx, vec, begin, end);
	}
	return merge_sort(ctx, vec, comp, begin, end);
}

bool TRTC_Sort(TRTCContext& ctx, DVVectorLike& vec, size_t begin, size_t end)
{
	Functor comp("Less");
	return TRTC_Sort(ctx, vec, comp, begin, end);
}

bool TRTC_Sort_By_Key(TRTCContext& ctx, DVVectorLike& keys, DVVectorLike& values, const Functor& comp, size_t begin_keys, size_t end_keys, size_t begin_values)
{
	if (comp.name_view_cls() == "Less")
	{
		std::string elem_type = keys.name_elem_cls();

		if (elem_type == "int8_t" || elem_type == "uint8_t" || elem_type == "int16_t" || elem_type == "uint16_t"
			|| elem_type == "int32_t" || elem_type == "uint32_t" || elem_type == "float")
			return radix_sort_by_key_32(ctx, keys, values, begin_keys, end_keys, begin_values);
		else if (elem_type == "int64_t" || elem_type == "uint64_t" || elem_type == "double")
			return radix_sort_by_key_64(ctx, keys, values, begin_keys, end_keys, begin_values);
	}
	else if (comp.name_view_cls() == "Greater")
	{
		std::string elem_type = keys.name_elem_cls();
		if (elem_type == "int8_t" || elem_type == "uint8_t" || elem_type == "int16_t" || elem_type == "uint16_t"
			|| elem_type == "int32_t" || elem_type == "uint32_t" || elem_type == "float")
			return radix_sort_by_key_reverse_32(ctx, keys, values, begin_keys, end_keys, begin_values);
		else if (elem_type == "int64_t" || elem_type == "uint64_t" || elem_type == "double")
			return radix_sort_by_key_reverse_64(ctx, keys, values, begin_keys, end_keys, begin_values);
	}
	return merge_sort_by_key(ctx, keys, values, comp, begin_keys, end_keys, begin_values);
}

bool TRTC_Sort_By_Key(TRTCContext& ctx, DVVectorLike& keys, DVVectorLike& values, size_t begin_keys , size_t end_keys, size_t begin_values)
{
	Functor comp("Less");
	return TRTC_Sort_By_Key(ctx, keys, values, comp, begin_keys, end_keys, begin_values);
}

bool TRTC_Is_Sorted(TRTCContext& ctx, const DVVectorLike& vec, const Functor& comp, bool& result, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	if (end <= begin + 1)
	{
		result = true;
		return true;
	}
	static TRTC_For s_for({ "vec", "comp", "res" }, "idx",
		"    if (comp(vec[idx+1], vec[idx])) res[0] = false;\n");

	result = true;
	DVVector dvres(ctx, "bool", 1, &result);
	const DeviceViewable* args[] = { &vec, &comp, &dvres };
	if (!s_for.launch(ctx, begin, end - 1, args)) return false;
	dvres.to_host(&result);
	return true;
}

bool TRTC_Is_Sorted(TRTCContext& ctx, const DVVectorLike& vec, bool& result, size_t begin, size_t end)
{
	Functor comp("Less");
	return TRTC_Is_Sorted(ctx, vec, comp, result, begin, end);
}

#include "general_find.h"

bool TRTC_Is_Sorted_Until(TRTCContext& ctx, const DVVectorLike& vec, const Functor& comp, size_t& result, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	size_t res_find = end - 1;
	if (end - 1 > begin)
	{
		Functor src(ctx, { {"vec", &vec}, {"comp", &comp} }, { "id" }, "        return comp(vec[id+1], vec[id]);\n");
		if (!general_find(ctx, begin, end - 1, src, res_find)) return false;
	}
	result = res_find == (size_t)(-1) ? end : res_find + 1;
	return true;
}

bool TRTC_Is_Sorted_Until(TRTCContext& ctx, const DVVectorLike& vec, size_t& result, size_t begin, size_t end)
{
	Functor comp("Less");
	return TRTC_Is_Sorted_Until(ctx, vec, comp, result, begin, end);
}
