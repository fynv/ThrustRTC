#include "merge.h"

bool TRTC_Merge(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, DVVectorLike& vec_out, const Functor& comp, size_t begin1, size_t end1, size_t begin2, size_t end2, size_t begin_out)
{
	if (end1 == (size_t)(-1)) end1 = vec1.size();
	if (end2 == (size_t)(-1)) end2 = vec2.size();

	static TRTC_For s_for(
		{ "vec1", "vec2",  "vec_out", "comp", "begin1", "end1", "begin2", "end2", "begin_out" }, "idx",
		"    int id1 = idx + begin1;\n"
		"    if (id1<end1)\n"
		"    {\n"
		"        size_t pos = d_lower_bound(vec2, vec1[id1], comp, begin2, end2);\n"
		"        vec_out[begin_out + idx + pos - begin2] = vec1[id1];\n"
		"    }\n"
		"    int id2 = idx + begin2;\n"
		"    if (id2<end2)\n"
		"    {\n"
		"        size_t pos = d_upper_bound(vec1, vec2[id2], comp, begin1, end1);\n"
		"        vec_out[begin_out + idx + pos - begin1] = vec2[id2];\n"
		"    }\n"
	);

	DVSizeT dvbegin1(begin1);
	DVSizeT dvend1(end1);
	DVSizeT dvbegin2(begin2);
	DVSizeT dvend2(end2);
	DVSizeT dvbegin_out(begin_out);

	size_t n = end1 - begin1;
	if (n < end2 - begin2) n = end2 - begin2;

	const DeviceViewable* args[] = { &vec1, &vec2, &vec_out, &comp, &dvbegin1, &dvend1, &dvbegin2, &dvend2, &dvbegin_out };
	return s_for.launch_n(ctx, n, args);
}

bool TRTC_Merge(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, DVVectorLike& vec_out, size_t begin1, size_t end1, size_t begin2, size_t end2, size_t begin_out)
{
	Functor comp("Less");
	return TRTC_Merge(ctx, vec1, vec2, vec_out, comp, begin1, end1, begin2, end2, begin_out);
}

bool TRTC_Merge_By_Key(TRTCContext& ctx, const DVVectorLike& keys1, const DVVectorLike& keys2, const DVVectorLike& value1, const DVVectorLike& value2, DVVectorLike& keys_out, DVVectorLike& value_out, const Functor& comp, size_t begin_keys1 , size_t end_keys1, size_t begin_keys2, size_t end_keys2, size_t begin_value1, size_t begin_value2, size_t begin_keys_out, size_t begin_value_out)
{
	if (end_keys1 == (size_t)(-1)) end_keys1 = keys1.size();
	if (end_keys2 == (size_t)(-1)) end_keys2 = keys2.size();

	static TRTC_For s_for(
		{ "keys1", "keys2", "value1", "value2", "keys_out", "value_out", "comp", "begin_keys1", "end_keys1", "begin_keys2", "end_keys2", "begin_value1", "begin_value2", "begin_keys_out", "begin_value_out" }, "idx",
		"    int id1 = idx + begin_keys1;\n"
		"    if (id1<end_keys1)\n"
		"    {\n"
		"        size_t pos = d_lower_bound(keys2, keys1[id1], comp, begin_keys2, end_keys2);\n"
		"        keys_out[begin_keys_out + idx + pos - begin_keys2] = keys1[id1];\n"
		"        value_out[begin_value_out + idx +  pos - begin_keys2] = value1[idx + begin_value1];\n"
		"    }\n"
		"    int id2 = idx + begin_keys2;\n"
		"    if (id2<end_keys2)\n"
		"    {\n"
		"        size_t pos = d_upper_bound(keys1, keys2[id2], comp, begin_keys1, end_keys1);\n"
		"        keys_out[begin_keys_out + idx + pos - begin_keys1] = keys2[id2];\n"
		"        value_out[begin_value_out + idx +  pos - begin_keys1] = value2[idx + begin_value1];\n"
		"    }\n"
	);

	DVSizeT dvbegin_keys1(begin_keys1);
	DVSizeT dvend_keys1(end_keys1);
	DVSizeT dvbegin_keys2(begin_keys2);
	DVSizeT dvend_keys2(end_keys2);
	DVSizeT dvbegin_value1(begin_value1);
	DVSizeT dvbegin_value2(begin_value2);
	DVSizeT dvbegin_keys_out(begin_keys_out);
	DVSizeT dvbegin_value_out(begin_value_out);

	size_t n = end_keys1 - begin_keys1;
	if (n < end_keys2 - begin_keys2) n = end_keys2 - begin_keys2;

	const DeviceViewable* args[] = { &keys1, &keys2, &value1, &value2, &keys_out, &value_out, &comp, &dvbegin_keys1, &dvend_keys1, &dvbegin_keys2, &dvend_keys2, &dvbegin_value1, &dvbegin_value2, &dvbegin_keys_out, &dvbegin_value_out };
	return s_for.launch_n(ctx, n, args);
}

bool TRTC_Merge_By_Key(TRTCContext& ctx, const DVVectorLike& keys1, const DVVectorLike& keys2, const DVVectorLike& value1, const DVVectorLike& value2, DVVectorLike& keys_out, DVVectorLike& value_out, size_t begin_keys1, size_t end_keys1, size_t begin_keys2, size_t end_keys2, size_t begin_value1, size_t begin_value2, size_t begin_keys_out, size_t begin_value_out)
{
	Functor comp("Less");
	return TRTC_Merge_By_Key(ctx, keys1, keys2, value1, value2, keys_out, value_out, comp, begin_keys1, end_keys2, begin_keys2, end_keys2, begin_value1, begin_value2, begin_keys_out, begin_value_out);
}
