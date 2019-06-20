#include "sort.h"
#include "merge_sort.h"
#include "radix_sort.h"

#define BLOCK_SIZE 256

bool TRTC_Sort(DVVectorLike& vec, const Functor& comp)
{
	if (comp.name_view_cls() == "Less")
	{
		std::string elem_type = vec.name_elem_cls();

		if (elem_type == "int8_t" || elem_type == "uint8_t" || elem_type == "int16_t" || elem_type == "uint16_t"
			|| elem_type == "int32_t" || elem_type == "uint32_t" || elem_type == "float")
			return radix_sort_32(vec);
		else if  (elem_type == "int64_t" || elem_type == "uint64_t" || elem_type == "double")
			return radix_sort_64(vec);
	}
	else if (comp.name_view_cls() == "Greater")
	{
		std::string elem_type = vec.name_elem_cls();
		if (elem_type == "int8_t" || elem_type == "uint8_t" || elem_type == "int16_t" || elem_type == "uint16_t"
			|| elem_type == "int32_t" || elem_type == "uint32_t" || elem_type == "float")
			return radix_sort_reverse_32(vec);
		else if (elem_type == "int64_t" || elem_type == "uint64_t" || elem_type == "double")
			return radix_sort_reverse_64(vec);
	}
	return merge_sort(vec, comp);
}

bool TRTC_Sort(DVVectorLike& vec)
{
	Functor comp("Less");
	return TRTC_Sort(vec, comp);
}

bool TRTC_Sort_By_Key(DVVectorLike& keys, DVVectorLike& values, const Functor& comp)
{
	if (comp.name_view_cls() == "Less")
	{
		std::string elem_type = keys.name_elem_cls();

		if (elem_type == "int8_t" || elem_type == "uint8_t" || elem_type == "int16_t" || elem_type == "uint16_t"
			|| elem_type == "int32_t" || elem_type == "uint32_t" || elem_type == "float")
			return radix_sort_by_key_32(keys, values);
		else if (elem_type == "int64_t" || elem_type == "uint64_t" || elem_type == "double")
			return radix_sort_by_key_64(keys, values);
	}
	else if (comp.name_view_cls() == "Greater")
	{
		std::string elem_type = keys.name_elem_cls();
		if (elem_type == "int8_t" || elem_type == "uint8_t" || elem_type == "int16_t" || elem_type == "uint16_t"
			|| elem_type == "int32_t" || elem_type == "uint32_t" || elem_type == "float")
			return radix_sort_by_key_reverse_32(keys, values);
		else if (elem_type == "int64_t" || elem_type == "uint64_t" || elem_type == "double")
			return radix_sort_by_key_reverse_64(keys, values);
	}
	return merge_sort_by_key(keys, values, comp);
}

bool TRTC_Sort_By_Key(DVVectorLike& keys, DVVectorLike& values)
{
	Functor comp("Less");
	return TRTC_Sort_By_Key(keys, values, comp);
}

bool TRTC_Is_Sorted(const DVVectorLike& vec, const Functor& comp, bool& result)
{
	if (vec.size()<=1)
	{
		result = true;
		return true;
	}
	static TRTC_For s_for({ "vec", "comp", "res" }, "idx",
		"    if (comp(vec[idx+1], vec[idx])) res[0] = false;\n");

	result = true;
	DVVector dvres("bool", 1, &result);
	const DeviceViewable* args[] = { &vec, &comp, &dvres };
	if (!s_for.launch_n(vec.size() - 1, args)) return false;
	dvres.to_host(&result);
	return true;
}

bool TRTC_Is_Sorted(const DVVectorLike& vec, bool& result)
{
	Functor comp("Less");
	return TRTC_Is_Sorted(vec, comp, result);
}

#include "general_find.h"

bool TRTC_Is_Sorted_Until(const DVVectorLike& vec, const Functor& comp, size_t& result)
{
	size_t res_find = vec.size() - 1;
	if (vec.size()>1)
	{
		Functor src({ {"vec", &vec}, {"comp", &comp} }, { "id" }, "        return comp(vec[id+1], vec[id]);\n");
		if (!general_find(vec.size()-1, src, res_find)) return false;
	}
	result = res_find == (size_t)(-1) ? vec.size() : res_find + 1;
	return true;
}

bool TRTC_Is_Sorted_Until(const DVVectorLike& vec, size_t& result)
{
	Functor comp("Less");
	return TRTC_Is_Sorted_Until(vec, comp, result);
}
