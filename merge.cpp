#include "merge.h"

bool TRTC_Merge(const DVVectorLike& vec1, const DVVectorLike& vec2, DVVectorLike& vec_out, const Functor& comp)
{
	static TRTC_For s_for(
		{ "vec1", "vec2",  "vec_out", "comp" }, "idx",
		"    if (idx<vec1.size())\n"
		"    {\n"
		"        size_t pos = d_lower_bound(vec2, vec1[idx], comp);\n"
		"        vec_out[idx + pos] = vec1[idx];\n"
		"    }\n"
		"    if (idx<vec2.size())\n"
		"    {\n"
		"        size_t pos = d_upper_bound(vec1, vec2[idx], comp);\n"
		"        vec_out[idx + pos] = vec2[idx];\n"
		"    }\n"
	);

	size_t n = vec1.size();
	if (n < vec2.size()) n = vec2.size();

	const DeviceViewable* args[] = { &vec1, &vec2, &vec_out, &comp };
	return s_for.launch_n(n, args);
}

bool TRTC_Merge(const DVVectorLike& vec1, const DVVectorLike& vec2, DVVectorLike& vec_out)
{
	Functor comp("Less");
	return TRTC_Merge(vec1, vec2, vec_out, comp);
}

bool TRTC_Merge_By_Key(const DVVectorLike& keys1, const DVVectorLike& keys2, const DVVectorLike& value1, const DVVectorLike& value2, DVVectorLike& keys_out, DVVectorLike& value_out, const Functor& comp)
{
	static TRTC_For s_for(
		{ "keys1", "keys2", "value1", "value2", "keys_out", "value_out", "comp" }, "idx",
		"    if (idx<keys1.size())\n"
		"    {\n"
		"        size_t pos = d_lower_bound(keys2, keys1[idx], comp);\n"
		"        keys_out[idx + pos] = keys1[idx];\n"
		"        value_out[idx +  pos] = value1[idx];\n"
		"    }\n"
		"    if (idx<keys2.size())\n"
		"    {\n"
		"        size_t pos = d_upper_bound(keys1, keys2[idx], comp);\n"
		"        keys_out[idx + pos] = keys2[idx];\n"
		"        value_out[idx +  pos] = value2[idx];\n"
		"    }\n"
	);

	size_t n = keys1.size();
	if (n < keys2.size()) n = keys2.size();

	const DeviceViewable* args[] = { &keys1, &keys2, &value1, &value2, &keys_out, &value_out, &comp };
	return s_for.launch_n(n, args);
}

bool TRTC_Merge_By_Key(const DVVectorLike& keys1, const DVVectorLike& keys2, const DVVectorLike& value1, const DVVectorLike& value2, DVVectorLike& keys_out, DVVectorLike& value_out)
{
	Functor comp("Less");
	return TRTC_Merge_By_Key(keys1, keys2, value1, value2, keys_out, value_out, comp);
}
