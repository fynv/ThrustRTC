#include "cuda_wrapper.h"
#include "binary_search.h"

inline bool CheckCUresult(CUresult res, const char* name_call)
{
	if (res != CUDA_SUCCESS)
	{
		printf("%s failed with Error code: %u\n", name_call, res);
		const char *name = nullptr;
		const char *desc = nullptr;
		cuGetErrorName(res, &name);
		cuGetErrorString(res, &desc);
		if (name != nullptr)
		{
			printf("Error Name: %s\n", name);
		}
		if (desc != nullptr)
		{
			printf("Error Description: %s\n", desc);
		}
		return false;
	}
	return true;
}

bool TRTC_Lower_Bound(const DVVectorLike& vec, const DeviceViewable& value, const Functor& comp, size_t& result)
{
	if (vec.size()<=0)
	{
		result = 0;
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
		DVSizeT _dv_num_grps(vec.size());
		DVSizeT _dv_begin(0);
		DVSizeT _dv_size_grp(1);
		DVSizeT _dv_div_id((size_t)(-1));
		const DeviceViewable* _args[] = { &_dv_num_grps, &vec, &_dv_begin, &value, &comp, &dv_range_out, &_dv_size_grp, &_dv_div_id };
		s_kernel.calc_number_blocks(_args, 128, numBlocks);
	}

	size_t s_begin = 0;
	size_t s_end = vec.size();

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
			numBlocks = (int)((num_grps + 127) / 128);
		}

		DVSizeT dv_num_grps(num_grps);
		DVSizeT dv_begin(s_begin);
		DVSizeT dv_size_grp(size_grp);
		DVSizeT dv_div_id(div_id);

		h_range_out[0] = vec.size();
		h_range_out[1] = 0;
		if (!CheckCUresult(cuMemcpyHtoD((CUdeviceptr)dv_range_out.data(), h_range_out, sizeof(size_t) * 2), "cuMemcpyHtoD()")) return false;

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

bool TRTC_Lower_Bound(const DVVectorLike& vec, const DeviceViewable& value, size_t& result)
{
	return TRTC_Lower_Bound(vec, value, Functor("Less"), result);
}

bool TRTC_Upper_Bound(const DVVectorLike& vec, const DeviceViewable& value, const Functor& comp, size_t& result)
{
	if (vec.size() <= 0)
	{
		result = 0;
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
		DVSizeT _dv_num_grps(vec.size());
		DVSizeT _dv_begin(0);
		DVSizeT _dv_size_grp(1);
		DVSizeT _dv_div_id((size_t)(-1));
		const DeviceViewable* _args[] = { &_dv_num_grps, &vec, &_dv_begin, &value, &comp, &dv_range_out, &_dv_size_grp, &_dv_div_id };
		s_kernel.calc_number_blocks(_args, 128, numBlocks);
	}

	size_t s_begin = 0;
	size_t s_end = vec.size();
	
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
			numBlocks = (int)((num_grps + 127) / 128);
		}

		DVSizeT dv_num_grps(num_grps);
		DVSizeT dv_begin(s_begin);
		DVSizeT dv_size_grp(size_grp);
		DVSizeT dv_div_id(div_id);

		h_range_out[0] = vec.size();
		h_range_out[1] = 0;
		if (!CheckCUresult(cuMemcpyHtoD((CUdeviceptr)dv_range_out.data(), h_range_out, sizeof(size_t) * 2), "cuMemcpyHtoD()")) return false;

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

bool TRTC_Upper_Bound(const DVVectorLike& vec, const DeviceViewable& value, size_t& result)
{
	return TRTC_Upper_Bound(vec, value, Functor("Less"), result);
}


bool TRTC_Binary_Search(const DVVectorLike& vec, const DeviceViewable& value, const Functor& comp, bool& result)
{
	if (vec.size() <= 0)
	{
		result = 0;
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
		DVSizeT _dv_num_grps(vec.size());
		DVSizeT _dv_begin(0);
		DVSizeT _dv_size_grp(1);
		DVSizeT _dv_div_id((size_t)(-1));
		const DeviceViewable* _args[] = { &_dv_num_grps, &vec, &_dv_begin, &value, &comp, &dv_range_out, &_dv_size_grp, &_dv_div_id };
		s_kernel.calc_number_blocks(_args, 128, numBlocks);
	}
	size_t s_begin = 0;
	size_t s_end = vec.size();

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
			numBlocks = (int)((num_grps + 127) / 128);
		}

		DVSizeT dv_num_grps(num_grps);
		DVSizeT dv_begin(s_begin);
		DVSizeT dv_size_grp(size_grp);
		DVSizeT dv_div_id(div_id);

		h_range_out[0] = (size_t)(-1);
		h_range_out[1] = 0;
		h_range_out[2] = 0;
		if (!CheckCUresult(cuMemcpyHtoD((CUdeviceptr)dv_range_out.data(), h_range_out, sizeof(size_t) * 3), "cuMemcpyHtoD()")) return false;

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

bool TRTC_Binary_Search(const DVVectorLike& vec, const DeviceViewable& value, bool& result)
{
	return TRTC_Binary_Search(vec, value, Functor("Less"), result);
}

bool TRTC_Lower_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, const Functor& comp)
{
	static TRTC_For s_for({ "vec", "values", "result", "comp" }, "idx",
		"    auto value = values[idx];\n"
		"    result[idx] =  (decltype(result)::value_t) d_lower_bound(vec, value, comp);\n"
	);

	const DeviceViewable* args[] = { &vec, &values, &result, &comp };
	return s_for.launch_n(values.size(), args);
}

bool TRTC_Lower_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result)
{
	return TRTC_Lower_Bound_V(vec, values, result, Functor("Less"));
}

bool TRTC_Upper_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, const Functor& comp)
{
	static TRTC_For s_for({ "vec", "values", "result", "comp" }, "idx",
		"    auto value = values[idx];\n"
		"    result[idx] =  (decltype(result)::value_t) d_upper_bound(vec, value, comp);\n"
	);

	const DeviceViewable* args[] = { &vec, &values, &result, &comp };
	return s_for.launch_n(values.size(), args);
}

bool TRTC_Upper_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result)
{
	return TRTC_Upper_Bound_V(vec, values, result, Functor("Less"));
}

bool TRTC_Binary_Search_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, const Functor& comp)
{
	static TRTC_For s_for({ "vec", "values", "result", "comp" }, "idx",
		"    auto value = values[idx ];\n"
		"    result[idx] =  (decltype(result)::value_t) d_binary_search(vec, value, comp);\n"
	);

	const DeviceViewable* args[] = { &vec, &values, &result, &comp };
	return s_for.launch_n(values.size(), args);
}

bool TRTC_Binary_Search_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result)
{
	return TRTC_Binary_Search_V(vec, values, result, Functor("Less"));
}
