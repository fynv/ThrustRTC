#include "radix_sort.h"
#include "extrema.h"
#include "general_reduce.h"
#include "built_in.h"

#include "copy.h"
#include "general_scan.h"
#include "cuda_wrapper.h"
#include "fake_vectors/DVRange.h"

static bool s_bit_mask32(const DVVectorLike& vec, const DVVector& dv_min, uint32_t& bit_mask)
{
	Functor src({ {"vec", &vec}, {"v_min", &dv_min } }, { "idx" },
		"        uint32_t cur_v = d_u32(vec[idx]);\n"
		"        uint32_t min_v = d_u32(v_min[0]);\n"
		"        uint32_t diff = cur_v - min_v;\n"
		"        return diff;\n");

	static Functor op({ }, { "i1", "i2" },
		"        return i1|i2;\n");

	ViewBuf buf;
	if (!general_reduce(vec.size(), "uint32_t", src, op, buf)) return false;
	bit_mask = *(uint32_t*)buf.data();

	return true;
}

static bool s_bit_mask64(const DVVectorLike& vec, const DVVector& dv_min, uint64_t& bit_mask)
{
	Functor src({ {"vec", &vec}, {"v_min", &dv_min } }, { "idx" },
		"        uint64_t cur_v = d_u64(vec[idx]);\n"
		"        uint64_t min_v = d_u64(v_min[0]);\n"
		"        uint64_t diff = cur_v - min_v;\n"
		"        return diff;\n");

	static Functor op({ }, { "i1", "i2" },
		"        return i1|i2;\n");

	ViewBuf buf;
	if (!general_reduce(vec.size(), "uint64_t", src, op, buf)) return false;
	bit_mask = *(uint64_t*)buf.data();

	return true;
}

static bool s_partition_scan_32(size_t n, const DVVector& src, const DVVector& dv_min, int bit, DVVector& inds, uint32_t& count)
{
	static Functor plus({}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");

	DVInt32 dvbit(bit);
	Functor src_scan({ {"src", &src}, {"v_min", &dv_min }, {"bit", &dvbit} }, { "idx" },
		"        uint32_t cur_v = d_u32(src[idx]);\n"
		"        uint32_t min_v = d_u32(v_min[0]);\n"
		"        uint32_t diff = cur_v - min_v;\n"
		"        bool pred = (diff & (((uint32_t)1)<<bit)) == 0;\n"
		"        return pred ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");

	if (!general_scan(n, src_scan, inds, plus)) return false;

	Pair<uint32_t, uint32_t> sums;
	cuMemcpyDtoH(&sums, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + n - 1), sizeof(Pair<uint32_t, uint32_t>));
	count = sums.first;
	return true;
}

static bool s_partition_scan_reverse_32(size_t n, const DVVector& src, const DVVector& dv_min, int bit, DVVector& inds, uint32_t& count)
{
	static Functor plus({}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");

	DVInt32 dvbit(bit);
	Functor src_scan({ {"src", &src}, {"v_min", &dv_min }, {"bit", &dvbit} }, { "idx" },
		"        uint32_t cur_v = d_u32(src[idx]);\n"
		"        uint32_t min_v = d_u32(v_min[0]);\n"
		"        uint32_t diff = cur_v - min_v;\n"
		"        bool pred = (diff & (((uint32_t)1)<<bit)) != 0;\n"
		"        return pred ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");

	if (!general_scan(n, src_scan, inds, plus)) return false;

	Pair<uint32_t, uint32_t> sums;
	cuMemcpyDtoH(&sums, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + n - 1), sizeof(Pair<uint32_t, uint32_t>));
	count = sums.first;
	return true;
}

static bool s_partition_scan_64(size_t n, const DVVector& src, const DVVector& dv_min, int bit, DVVector& inds, uint32_t& count)
{
	static Functor plus({}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");

	DVInt32 dvbit(bit);
	Functor src_scan({ {"src", &src}, {"v_min", &dv_min }, {"bit", &dvbit} }, { "idx" },
		"        uint64_t cur_v = d_u64(src[idx]);\n"
		"        uint64_t min_v = d_u64(v_min[0]);\n"
		"        uint64_t diff = cur_v - min_v;\n"
		"        bool pred = (diff & (((uint64_t)1)<<bit)) == 0;\n"
		"        return pred ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");

	if (!general_scan(n, src_scan, inds, plus)) return false;

	Pair<uint32_t, uint32_t> sums;
	cuMemcpyDtoH(&sums, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + n - 1), sizeof(Pair<uint32_t, uint32_t>));
	count = sums.first;
	return true;
}

static bool s_partition_scan_reverse_64(size_t n, const DVVector& src, const DVVector& dv_min, int bit, DVVector& inds, uint32_t& count)
{
	static Functor plus({}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");

	DVInt32 dvbit(bit);
	Functor src_scan({ {"src", &src}, {"v_min", &dv_min }, {"bit", &dvbit} }, { "idx" },
		"        uint64_t cur_v = d_u64(src[idx]);\n"
		"        uint64_t min_v = d_u64(v_min[0]);\n"
		"        uint64_t diff = cur_v - min_v;\n"
		"        bool pred = (diff & (((uint64_t)1)<<bit)) != 0;\n"
		"        return pred ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");

	if (!general_scan(n, src_scan, inds, plus)) return false;

	Pair<uint32_t, uint32_t> sums;
	cuMemcpyDtoH(&sums, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + n - 1), sizeof(Pair<uint32_t, uint32_t>));
	count = sums.first;
	return true;
}

static bool s_partition_scatter(size_t n, const DVVector& src, const DVVector& inds, DVVectorLike& dst, uint32_t count)
{
	static TRTC_For s_for_scatter({ "src", "inds", "dst", "count" }, "idx",
		"    if ((idx==0 && inds[idx].first>0) || (idx>0 && inds[idx].first>inds[idx-1].first))\n"
		"        dst[inds[idx].first -1] = src[idx];\n"
		"    else\n"
		"        dst[count + inds[idx].second - 1] = src[idx];\n"
	);

	DVUInt32 dvcount(count);
	const DeviceViewable* args[] = { &src, &inds, &dst, &dvcount };
	return s_for_scatter.launch_n(n, args);
}

static bool s_partition_scatter_by_keys(size_t n, const DVVector& src_keys, const DVVector& src_values, const DVVector& inds, DVVectorLike& dst_keys, DVVectorLike& dst_values, uint32_t count)
{
	static TRTC_For s_for_scatter({ "src_keys", "src_values", "inds", "dst_keys", "dst_values", "count" }, "idx",
		"    if ((idx==0 && inds[idx].first>0) || (idx>0 && inds[idx].first>inds[idx-1].first))\n"
		"    {\n"
		"        dst_keys[inds[idx].first -1] = src_keys[idx];\n"
		"        dst_values[inds[idx].first -1] = src_values[idx];\n"
		"    }\n"
		"    else\n"
		"    {\n"
		"        dst_keys[count + inds[idx].second - 1] = src_keys[idx];\n"
		"        dst_values[count + inds[idx].second - 1] = src_values[idx];\n"
		"    }\n"
	);

	DVUInt32 dvcount(count);
	const DeviceViewable* args[] = { &src_keys, &src_values, &inds, &dst_keys, &dst_values, &dvcount };
	return s_for_scatter.launch_n(n, args);
}

bool radix_sort_32(DVVectorLike& vec)
{
	size_t id_min;
	if (!TRTC_Min_Element(vec, id_min)) return false;

	DVVector dv_min(vec.name_elem_cls().c_str(), 1);
	TRTC_Copy(DVRange(vec, id_min, id_min+1), dv_min);

	uint32_t bit_mask;
	if (!s_bit_mask32(vec, dv_min, bit_mask)) return false;
	if (bit_mask == 0) return true; // already sorted

	size_t n = vec.size();
	DVVector tmp1(vec.name_elem_cls().c_str(), n);
	DVVector tmp2(vec.name_elem_cls().c_str(), n);
	if (!TRTC_Copy(vec, tmp1)) return false;
	DVVector* p1 = &tmp1;
	DVVector* p2 = &tmp2;
	DVVector inds("Pair<uint32_t, uint32_t>", n);
	for (int bit = 0; bit <32; bit++)
	{
		if ( (bit_mask & (((uint32_t)1) << bit)) == 0) continue;
		
		uint32_t count;
		if (!s_partition_scan_32(n, *p1, dv_min, bit, inds, count)) return false;
		if (!s_partition_scatter(n, *p1, inds, *p2, count)) return false;

		{
			DVVector* tmp = p1;
			p1 = p2;
			p2 = tmp;
		}
	}
	if (!TRTC_Copy(*p1, vec)) return false;
	return true;
}

bool radix_sort_reverse_32(DVVectorLike& vec)
{
	size_t id_min;
	if (!TRTC_Min_Element(vec, id_min)) return false;

	DVVector dv_min(vec.name_elem_cls().c_str(), 1);
	TRTC_Copy(DVRange(vec, id_min, id_min + 1), dv_min);

	uint32_t bit_mask;
	if (!s_bit_mask32(vec, dv_min, bit_mask)) return false;
	if (bit_mask == 0) return true; // already sorted

	size_t n = vec.size();
	DVVector tmp1(vec.name_elem_cls().c_str(), n);
	DVVector tmp2(vec.name_elem_cls().c_str(), n);
	if (!TRTC_Copy(vec, tmp1)) return false;
	DVVector* p1 = &tmp1;
	DVVector* p2 = &tmp2;
	DVVector inds("Pair<uint32_t, uint32_t>", n);
	for (int bit = 0; bit < 32; bit++)
	{
		if ((bit_mask & (((uint32_t)1) << bit)) == 0) continue;

		uint32_t count;
		if (!s_partition_scan_reverse_32(n, *p1, dv_min, bit, inds, count)) return false;
		if (!s_partition_scatter(n, *p1, inds, *p2, count)) return false;

		{
			DVVector* tmp = p1;
			p1 = p2;
			p2 = tmp;
		}
	}
	if (!TRTC_Copy(*p1, vec)) return false;
	return true;
}

bool radix_sort_64(DVVectorLike& vec)
{
	size_t id_min;
	if (!TRTC_Min_Element(vec, id_min)) return false;

	DVVector dv_min(vec.name_elem_cls().c_str(), 1);
	TRTC_Copy(DVRange(vec, id_min, id_min + 1), dv_min);

	uint64_t bit_mask;
	if (!s_bit_mask64(vec, dv_min, bit_mask)) return false;
	if (bit_mask == 0) return true; // already sorted

	size_t n = vec.size();
	DVVector tmp1(vec.name_elem_cls().c_str(), n);
	DVVector tmp2(vec.name_elem_cls().c_str(), n);
	if (!TRTC_Copy(vec, tmp1)) return false;
	DVVector* p1 = &tmp1;
	DVVector* p2 = &tmp2;
	DVVector inds("Pair<uint32_t, uint32_t>", n);
	
	for (int bit = 0; bit < 64; bit++)
	{
		if ((bit_mask & (((uint64_t)1) << bit)) == 0) continue;

		uint32_t count;
		if (!s_partition_scan_64(n, *p1, dv_min, bit, inds, count)) return false;
		if (!s_partition_scatter(n, *p1, inds, *p2, count)) return false;

		{
			DVVector* tmp = p1;
			p1 = p2;
			p2 = tmp;
		}
	}
	if (!TRTC_Copy(*p1, vec)) return false;
	return true;
}

bool radix_sort_reverse_64(DVVectorLike& vec)
{
	size_t id_min;
	if (!TRTC_Min_Element(vec, id_min)) return false;

	DVVector dv_min(vec.name_elem_cls().c_str(), 1);
	TRTC_Copy(DVRange(vec, id_min, id_min + 1), dv_min);

	uint64_t bit_mask;
	if (!s_bit_mask64(vec, dv_min, bit_mask)) return false;
	if (bit_mask == 0) return true; // already sorted

	size_t n = vec.size();
	DVVector tmp1(vec.name_elem_cls().c_str(), n);
	DVVector tmp2(vec.name_elem_cls().c_str(), n);
	if (!TRTC_Copy(vec, tmp1)) return false;
	DVVector* p1 = &tmp1;
	DVVector* p2 = &tmp2;
	DVVector inds("Pair<uint32_t, uint32_t>", n);

	for (int bit = 0; bit < 64; bit++)
	{
		if ((bit_mask & (((uint64_t)1) << bit)) == 0) continue;

		uint32_t count;
		if (!s_partition_scan_reverse_64(n, *p1, dv_min, bit, inds, count)) return false;
		if (!s_partition_scatter(n, *p1, inds, *p2, count)) return false;

		{
			DVVector* tmp = p1;
			p1 = p2;
			p2 = tmp;
		}
	}
	if (!TRTC_Copy(*p1, vec)) return false;
	return true;
}

bool radix_sort_by_key_32(DVVectorLike& keys, DVVectorLike& values)
{
	size_t id_min;
	if (!TRTC_Min_Element(keys, id_min)) return false;

	DVVector dv_min(keys.name_elem_cls().c_str(), 1);
	TRTC_Copy(DVRange(keys, id_min, id_min + 1), dv_min);

	uint32_t bit_mask;
	if (!s_bit_mask32(keys, dv_min, bit_mask)) return false;
	if (bit_mask == 0) return true; // already sorted

	size_t n_keys = keys.size();
	size_t n_values = keys.size();
	DVVector tmp_keys1(keys.name_elem_cls().c_str(), n_keys);
	DVVector tmp_keys2(keys.name_elem_cls().c_str(), n_keys);
	DVVector tmp_values1(values.name_elem_cls().c_str(), n_keys);
	DVVector tmp_values2(values.name_elem_cls().c_str(), n_keys);
	
	if (!TRTC_Copy(keys, tmp_keys1)) return false;

	if (n_keys == n_values)
	{
		if (!TRTC_Copy(values, tmp_values1)) return false;
	}
	else
	{
		if (!TRTC_Copy(DVRange(values, 0, n_keys), tmp_values1)) return false;
	}

	DVVector* p_keys1 = &tmp_keys1;
	DVVector* p_keys2 = &tmp_keys2;
	DVVector* p_values1 = &tmp_values1;
	DVVector* p_values2 = &tmp_values2;
	DVVector inds("Pair<uint32_t, uint32_t>", n_keys);

	for (int bit = 0; bit < 32; bit++)
	{
		if ((bit_mask & (((uint32_t)1) << bit)) == 0) continue;

		uint32_t count;
		if (!s_partition_scan_32(n_keys, *p_keys1, dv_min, bit, inds, count)) return false;
		if (!s_partition_scatter_by_keys(n_keys, *p_keys1, *p_values1, inds, *p_keys2, *p_values2, count)) return false;

		{
			DVVector* tmp = p_keys1;
			p_keys1 = p_keys2;
			p_keys2 = tmp;

			tmp = p_values1;
			p_values1 = p_values2;
			p_values2 = tmp;
		}
	}
	if (!TRTC_Copy(*p_keys1, keys)) return false;
	if (!TRTC_Copy(*p_values1, values)) return false;

	return true;
}

bool radix_sort_by_key_reverse_32(DVVectorLike& keys, DVVectorLike& values)
{
	size_t id_min;
	if (!TRTC_Min_Element(keys, id_min)) return false;

	DVVector dv_min(keys.name_elem_cls().c_str(), 1);
	TRTC_Copy(DVRange(keys, id_min, id_min + 1), dv_min);

	uint32_t bit_mask;
	if (!s_bit_mask32(keys, dv_min, bit_mask)) return false;
	if (bit_mask == 0) return true; // already sorted

	size_t n_keys = keys.size();
	size_t n_values = keys.size();
	DVVector tmp_keys1(keys.name_elem_cls().c_str(), n_keys);
	DVVector tmp_keys2(keys.name_elem_cls().c_str(), n_keys);
	DVVector tmp_values1(values.name_elem_cls().c_str(), n_keys);
	DVVector tmp_values2(values.name_elem_cls().c_str(), n_keys);
	
	if (!TRTC_Copy(keys, tmp_keys1)) return false;

	if (n_keys == n_values)
	{
		if (!TRTC_Copy(values, tmp_values1)) return false;
	}
	else
	{
		if (!TRTC_Copy(DVRange(values, 0, n_keys), tmp_values1)) return false;
	}

	DVVector* p_keys1 = &tmp_keys1;
	DVVector* p_keys2 = &tmp_keys2;
	DVVector* p_values1 = &tmp_values1;
	DVVector* p_values2 = &tmp_values2;
	DVVector inds("Pair<uint32_t, uint32_t>", n_keys);

	for (int bit = 0; bit < 32; bit++)
	{
		if ((bit_mask & (((uint32_t)1) << bit)) == 0) continue;

		uint32_t count;
		if (!s_partition_scan_reverse_32(n_keys, *p_keys1, dv_min, bit, inds, count)) return false;
		if (!s_partition_scatter_by_keys(n_keys, *p_keys1, *p_values1, inds, *p_keys2, *p_values2, count)) return false;

		{
			DVVector* tmp = p_keys1;
			p_keys1 = p_keys2;
			p_keys2 = tmp;

			tmp = p_values1;
			p_values1 = p_values2;
			p_values2 = tmp;
		}
	}
	if (!TRTC_Copy(*p_keys1, keys)) return false;
	if (!TRTC_Copy(*p_values1, values)) return false;

	return true;
}

bool radix_sort_by_key_64(DVVectorLike& keys, DVVectorLike& values)
{
	size_t id_min;
	if (!TRTC_Min_Element(keys, id_min)) return false;

	DVVector dv_min(keys.name_elem_cls().c_str(), 1);
	TRTC_Copy(DVRange(keys, id_min, id_min + 1), dv_min);

	uint64_t bit_mask;
	if (!s_bit_mask64(keys, dv_min, bit_mask)) return false;
	if (bit_mask == 0) return true; // already sorted

	size_t n_keys = keys.size();
	size_t n_values = keys.size();
	DVVector tmp_keys1(keys.name_elem_cls().c_str(), n_keys);
	DVVector tmp_keys2(keys.name_elem_cls().c_str(), n_keys);
	DVVector tmp_values1(values.name_elem_cls().c_str(), n_keys);
	DVVector tmp_values2(values.name_elem_cls().c_str(), n_keys);

	if (!TRTC_Copy(keys, tmp_keys1)) return false;

	if (n_keys == n_values)
	{
		if (!TRTC_Copy(values, tmp_values1)) return false;
	}
	else
	{
		if (!TRTC_Copy(DVRange(values, 0, n_keys), tmp_values1)) return false;
	}

	DVVector* p_keys1 = &tmp_keys1;
	DVVector* p_keys2 = &tmp_keys2;
	DVVector* p_values1 = &tmp_values1;
	DVVector* p_values2 = &tmp_values2;
	DVVector inds("Pair<uint32_t, uint32_t>", n_keys);

	for (int bit = 0; bit < 64; bit++)
	{
		if ((bit_mask & (((uint64_t)1) << bit)) == 0) continue;
	
		uint32_t count;
		if (!s_partition_scan_64(n_keys, *p_keys1, dv_min, bit, inds, count)) return false;
		if (!s_partition_scatter_by_keys(n_keys, *p_keys1, *p_values1, inds, *p_keys2, *p_values2, count)) return false;

		{
			DVVector* tmp = p_keys1;
			p_keys1 = p_keys2;
			p_keys2 = tmp;

			tmp = p_values1;
			p_values1 = p_values2;
			p_values2 = tmp;
		}
	}
	if (!TRTC_Copy(*p_keys1, keys)) return false;
	if (!TRTC_Copy(*p_values1, values)) return false;

	return true;
}

bool radix_sort_by_key_reverse_64(DVVectorLike& keys, DVVectorLike& values)
{
	size_t id_min;
	if (!TRTC_Min_Element(keys, id_min)) return false;

	DVVector dv_min(keys.name_elem_cls().c_str(), 1);
	TRTC_Copy(DVRange(keys, id_min, id_min + 1), dv_min);

	uint64_t bit_mask;
	if (!s_bit_mask64(keys, dv_min, bit_mask)) return false;
	if (bit_mask == 0) return true; // already sorted

	size_t n_keys = keys.size();
	size_t n_values = keys.size();
	DVVector tmp_keys1(keys.name_elem_cls().c_str(), n_keys);
	DVVector tmp_keys2(keys.name_elem_cls().c_str(), n_keys);
	DVVector tmp_values1(values.name_elem_cls().c_str(), n_keys);
	DVVector tmp_values2(values.name_elem_cls().c_str(), n_keys);

	if (!TRTC_Copy(keys, tmp_keys1)) return false;

	if (n_keys == n_values)
	{
		if (!TRTC_Copy(values, tmp_values1)) return false;
	}
	else
	{
		if (!TRTC_Copy(DVRange(values, 0, n_keys), tmp_values1)) return false;
	}

	DVVector* p_keys1 = &tmp_keys1;
	DVVector* p_keys2 = &tmp_keys2;
	DVVector* p_values1 = &tmp_values1;
	DVVector* p_values2 = &tmp_values2;
	DVVector inds("Pair<uint32_t, uint32_t>", n_keys);

	for (int bit = 0; bit < 64; bit++)
	{
		if ((bit_mask & (((uint64_t)1) << bit)) == 0) continue;

		uint32_t count;
		if (!s_partition_scan_reverse_64(n_keys, *p_keys1, dv_min, bit, inds, count)) return false;
		if (!s_partition_scatter_by_keys(n_keys, *p_keys1, *p_values1, inds, *p_keys2, *p_values2, count)) return false;

		{
			DVVector* tmp = p_keys1;
			p_keys1 = p_keys2;
			p_keys2 = tmp;

			tmp = p_values1;
			p_values1 = p_values2;
			p_values2 = tmp;
		}
	}
	if (!TRTC_Copy(*p_keys1, keys)) return false;
	if (!TRTC_Copy(*p_values1, values)) return false;

	return true;
}



