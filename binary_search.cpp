#include "cuda_wrapper.h"
#include "binary_search.h"

bool TRTC_Lower_Bound(const DVVectorLike& vec, const DeviceViewable& value, const Functor& comp, size_t& result, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	if (end <= begin)
	{
		result = begin;
		return true;
	}

	static TRTC_Kernel s_kernel(
		{ "num_grps", "vec", "begin", "value", "comp", "range_out", "size_grp", "div_id" },
		"    size_t id = threadIdx.x+blockIdx.x*blockDim.x;\n"
		"    if (id>=num_grps) return;"
		"    size_t begin_grp = size_grp*id + begin;\n"
		"    size_t end_grp = begin_grp + size_grp;\n"
		"    if (id>=div_id)\n"
		"    {\n"
		"        begin_grp += id - div_id;\n"
		"        end_grp = begin_grp + size_grp + 1;\n"
		"    }\n"
		"    if ( (id == 0 || comp(vec[begin_grp-1],value)) && !comp(vec[end_grp-1], value) )\n"
		"    {\n"
		"        range_out[0] = begin_grp;\n"
		"        range_out[1] = end_grp;\n"
		"    }\n"
	);

	size_t h_range_out[2];
	DVVector dv_range_out("size_t", 2);
	int numBlocks;
	{
		DVSizeT _dv_num_grps(end - begin);
		DVSizeT _dv_begin(begin);
		DVSizeT _dv_size_grp(1);
		DVSizeT _dv_div_id((size_t)(-1));
		const DeviceViewable* _args[] = { &_dv_num_grps, &vec, &_dv_begin, &value, &comp, &dv_range_out, &_dv_size_grp, &_dv_div_id };
		s_kernel.calc_number_blocks(_args, 128, numBlocks);
	}

	size_t s_begin = begin;
	size_t s_end = end;

	do
	{
		size_t n = s_end - s_begin;
		size_t size_grp = 1;
		size_t div_id = (size_t)(-1);
		size_t num_grps = 128 * numBlocks;
		if (num_grps < n)
		{
			size_grp = n / num_grps;
			div_id = (size_grp + 1) * num_grps - n;
		}
		else
		{
			num_grps = n;
			numBlocks = (num_grps + 127) / 128;
		}

		DVSizeT dv_num_grps(num_grps);
		DVSizeT dv_begin(s_begin);
		DVSizeT dv_size_grp(size_grp);
		DVSizeT dv_div_id(div_id);

		h_range_out[0] = end;
		h_range_out[1] = begin;
		cuMemcpyHtoD((CUdeviceptr)dv_range_out.data(), h_range_out, sizeof(size_t) * 2);

		const DeviceViewable* args[] = { &dv_num_grps, &vec, &dv_begin, &value, &comp, &dv_range_out, &dv_size_grp, &dv_div_id };
		if (!s_kernel.launch({ (unsigned)numBlocks, 1,1 }, { 128, 1, 1 }, args)) return false;
		dv_range_out.to_host(h_range_out);
		s_begin = h_range_out[0];
		s_end = h_range_out[1];
	}
	while (s_end > 0 && s_end > s_begin + 1);

	result = s_begin;
	return true;
}

bool TRTC_Lower_Bound(const DVVectorLike& vec, const DeviceViewable& value, size_t& result, size_t begin, size_t end)
{
	return TRTC_Lower_Bound(vec, value, Functor("Less"), result, begin, end);
}

bool TRTC_Upper_Bound(const DVVectorLike& vec, const DeviceViewable& value, const Functor& comp, size_t& result, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	if (end <= begin)
	{
		result = end;
		return true;
	}

	static TRTC_Kernel s_kernel(
		{ "num_grps", "vec", "begin", "value", "comp", "range_out", "size_grp", "div_id" },
		"    size_t id = threadIdx.x+blockIdx.x*blockDim.x;\n"
		"    if (id>=num_grps) return;"
		"    size_t begin_grp = size_grp*id + begin;\n"
		"    size_t end_grp = begin_grp + size_grp;\n"
		"    if (id>=div_id)\n"
		"    {\n"
		"        begin_grp += id - div_id;\n"
		"        end_grp = begin_grp + size_grp + 1;\n"
		"    }\n"
		"    if ( !comp(value, vec[begin_grp]) && (id == num_grps - 1 || comp(value, vec[end_grp]) ) )\n"
		"    {\n"
		"        range_out[0] = begin_grp;\n"
		"        range_out[1] = end_grp;\n"
		"    }\n"
	);

	size_t h_range_out[2];
	DVVector dv_range_out("size_t", 2);
	int numBlocks;
	{
		DVSizeT _dv_num_grps(end - begin);
		DVSizeT _dv_begin(begin);
		DVSizeT _dv_size_grp(1);
		DVSizeT _dv_div_id((size_t)(-1));
		const DeviceViewable* _args[] = { &_dv_num_grps, &vec, &_dv_begin, &value, &comp, &dv_range_out, &_dv_size_grp, &_dv_div_id };
		s_kernel.calc_number_blocks(_args, 128, numBlocks);
	}

	size_t s_begin = begin;
	size_t s_end = end;
	
	do
	{
		size_t n = s_end - s_begin;
		size_t size_grp = 1;
		size_t div_id = (size_t)(-1);
		size_t num_grps = 128 * numBlocks;
		if (num_grps < n)
		{
			size_grp = n / num_grps;
			div_id = (size_grp + 1) * num_grps - n;
		}
		else
		{
			num_grps = n;
			numBlocks = (num_grps + 127) / 128;
		}

		DVSizeT dv_num_grps(num_grps);
		DVSizeT dv_begin(s_begin);
		DVSizeT dv_size_grp(size_grp);
		DVSizeT dv_div_id(div_id);

		h_range_out[0] = end;
		h_range_out[1] = begin;
		cuMemcpyHtoD((CUdeviceptr)dv_range_out.data(), h_range_out, sizeof(size_t) * 2);

		const DeviceViewable* args[] = { &dv_num_grps, &vec, &dv_begin, &value, &comp, &dv_range_out, &dv_size_grp, &dv_div_id };
		if (!s_kernel.launch({ (unsigned)numBlocks, 1,1 }, { 128, 1, 1 }, args)) return false;
		dv_range_out.to_host(h_range_out);
		s_begin = h_range_out[0];
		s_end = h_range_out[1];
	}
	while (s_end > 0 && s_end > s_begin + 1);

	result = s_end;
	return true;
}

bool TRTC_Upper_Bound(const DVVectorLike& vec, const DeviceViewable& value, size_t& result, size_t begin, size_t end)
{
	return TRTC_Upper_Bound(vec, value, Functor("Less"), result, begin, end);
}


bool TRTC_Binary_Search(const DVVectorLike& vec, const DeviceViewable& value, const Functor& comp, bool& result, size_t begin, size_t end)
{
	if (end == (size_t)(-1)) end = vec.size();
	if (end <= begin)
	{
		result = false;
		return true;
	}

	static TRTC_Kernel s_kernel(
		{ "num_grps", "vec", "begin", "value", "comp", "range_out", "size_grp", "div_id" },
		"    size_t id = threadIdx.x+blockIdx.x*blockDim.x;\n"
		"    if (id>=num_grps) return;"
		"    size_t begin_grp = size_grp*id + begin;\n"
		"    size_t end_grp = begin_grp + size_grp;\n"
		"    if (id>=div_id)\n"
		"    {\n"
		"        begin_grp += id - div_id;\n"
		"        end_grp = begin_grp + size_grp + 1;\n"
		"    }\n"
		"    if ( !comp(value, vec[begin_grp]) && !comp(vec[end_grp-1], value) )\n"
		"    {\n"
		"        if (!comp(vec[begin_grp], value) || !comp(value, vec[end_grp-1]))\n"
		"        {\n"
		"              range_out[2] = 1;\n"
		"              return;"
		"        }\n"
		"        range_out[0] = begin_grp;\n"
		"        range_out[1] = end_grp;\n"
		"    }\n"
	);

	size_t h_range_out[3];
	DVVector dv_range_out("size_t", 3);
	int numBlocks;
	{
		DVSizeT _dv_num_grps(end - begin);
		DVSizeT _dv_begin(begin);
		DVSizeT _dv_size_grp(1);
		DVSizeT _dv_div_id((size_t)(-1));
		const DeviceViewable* _args[] = { &_dv_num_grps, &vec, &_dv_begin, &value, &comp, &dv_range_out, &_dv_size_grp, &_dv_div_id };
		s_kernel.calc_number_blocks(_args, 128, numBlocks);
	}
	size_t s_begin = begin;
	size_t s_end = end;

	do
	{
		size_t n = s_end - s_begin;
		size_t size_grp = 1;
		size_t div_id = (size_t)(-1);
		size_t num_grps = 128 * numBlocks;
		if (num_grps < n)
		{
			size_grp = n / num_grps;
			div_id = (size_grp + 1) * num_grps - n;
		}
		else
		{
			num_grps = n;
			numBlocks = (num_grps + 127) / 128;
		}

		DVSizeT dv_num_grps(num_grps);
		DVSizeT dv_begin(s_begin);
		DVSizeT dv_size_grp(size_grp);
		DVSizeT dv_div_id(div_id);

		h_range_out[0] = (size_t)(-1);
		h_range_out[1] = 0;
		h_range_out[2] = 0;
		cuMemcpyHtoD((CUdeviceptr)dv_range_out.data(), h_range_out, sizeof(size_t) * 3);

		const DeviceViewable* args[] = { &dv_num_grps, &vec, &dv_begin, &value, &comp, &dv_range_out, &dv_size_grp, &dv_div_id };
		if (!s_kernel.launch({ (unsigned)numBlocks, 1,1 }, { 128, 1, 1 }, args)) return false;
		dv_range_out.to_host(h_range_out);
		if (h_range_out[2] != 0) break;
		s_begin = h_range_out[0];
		s_end = h_range_out[1];

	} while (s_end - s_begin > 1);

	result = h_range_out[2] != 0;
	return true;

}

bool TRTC_Binary_Search(const DVVectorLike& vec, const DeviceViewable& value, bool& result, size_t begin, size_t end)
{
	return TRTC_Binary_Search(vec, value, Functor("Less"), result, begin, end);
}

bool TRTC_Lower_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, const Functor& comp, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
{
	if (end == (size_t)(-1)) end = vec.size();
	if (end_values == (size_t)(-1)) end_values = values.size();

	static TRTC_For s_for({ "vec", "values", "result", "begin", "end", "begin_values", "begin_result", "comp" }, "idx",
		"    auto value = values[idx + begin_values];\n"
		"    result[idx + begin_result] =  (decltype(result)::value_t) d_lower_bound(vec, value, comp, begin, end);\n"
	);

	DVSizeT dvbegin(begin);
	DVSizeT dvend(end);
	DVSizeT dvbegin_values(begin_values);
	DVSizeT dvbegin_result(begin_result);

	const DeviceViewable* args[] = { &vec, &values, &result, &dvbegin, &dvend, &dvbegin_values, &dvbegin_result, &comp };
	return s_for.launch_n(end_values-begin_values, args);
}

bool TRTC_Lower_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
{
	return TRTC_Lower_Bound_V(vec, values, result, Functor("Less"), begin, end, begin_values, end_values, begin_result);
}

bool TRTC_Upper_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, const Functor& comp, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
{
	if (end == (size_t)(-1)) end = vec.size();
	if (end_values == (size_t)(-1)) end_values = values.size();

	static TRTC_For s_for({ "vec", "values", "result", "begin", "end", "begin_values", "begin_result", "comp" }, "idx",
		"    auto value = values[idx + begin_values];\n"
		"    result[idx + begin_result] =  (decltype(result)::value_t) d_upper_bound(vec, value, comp, begin, end);\n"
	);

	DVSizeT dvbegin(begin);
	DVSizeT dvend(end);
	DVSizeT dvbegin_values(begin_values);
	DVSizeT dvbegin_result(begin_result);

	const DeviceViewable* args[] = { &vec, &values, &result, &dvbegin, &dvend, &dvbegin_values, &dvbegin_result, &comp };
	return s_for.launch_n(end_values - begin_values, args);
}

bool TRTC_Upper_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
{
	return TRTC_Upper_Bound_V(vec, values, result, Functor("Less"), begin, end, begin_values, end_values, begin_result);
}


bool TRTC_Binary_Search_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, const Functor& comp, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
{
	if (end == (size_t)(-1)) end = vec.size();
	if (end_values == (size_t)(-1)) end_values = values.size();

	static TRTC_For s_for({ "vec", "values", "result", "begin", "end", "begin_values", "begin_result", "comp" }, "idx",
		"    auto value = values[idx + begin_values];\n"
		"    result[idx + begin_result] =  (decltype(result)::value_t) d_binary_search(vec, value, comp, begin, end);\n"
	);

	DVSizeT dvbegin(begin);
	DVSizeT dvend(end);
	DVSizeT dvbegin_values(begin_values);
	DVSizeT dvbegin_result(begin_result);

	const DeviceViewable* args[] = { &vec, &values, &result, &dvbegin, &dvend, &dvbegin_values, &dvbegin_result, &comp };
	return s_for.launch_n(end_values - begin_values, args);
}

bool TRTC_Binary_Search_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, size_t begin, size_t end, size_t begin_values, size_t end_values, size_t begin_result)
{
	return TRTC_Binary_Search_V(vec, values, result, Functor("Less"), begin, end, begin_values, end_values, begin_result);
}
