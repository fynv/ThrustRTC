#include "copy.h"
#include "partition.h"
#include "general_scan.h"
#include "cuda_wrapper.h"
#include "built_in.h"

uint32_t TRTC_Partition(DVVectorLike& vec, const Functor& pred)
{
	DVVector cpy(vec.name_elem_cls().c_str(), vec.size());
	TRTC_Copy(vec, cpy);

	Functor src_scan({ {"src", &cpy}, { "pred", &pred } }, { "idx" },
		"        return pred(src[idx]) ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");
	DVVector inds("Pair<uint32_t, uint32_t>", vec.size());
	Functor plus({}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");
	if (!general_scan(vec.size(), src_scan, inds, plus)) return (uint32_t)(-1);
	Pair<uint32_t, uint32_t> ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + vec.size() - 1), sizeof(Pair<uint32_t, uint32_t>));

	static TRTC_For s_for_scatter({ "vec_cpy", "inds", "vec", "count" }, "idx",
		"    if ((idx==0 && inds[idx].first>0) || (idx>0 && inds[idx].first>inds[idx-1].first))\n"
		"        vec[inds[idx].first -1] = vec_cpy[idx];\n"
		"    else\n"
		"        vec[count + inds[idx].second - 1] = vec_cpy[idx];\n"
	);
	DVUInt32 dvcount(ret.first);
	const DeviceViewable* args[] = { &cpy, &inds, &vec, &dvcount };
	if (!s_for_scatter.launch_n(vec.size(), args)) return (uint32_t)(-1);
	return ret.first;
}

uint32_t TRTC_Partition_Stencil(DVVectorLike& vec, const DVVectorLike& stencil, const Functor& pred)
{
	DVVector cpy(vec.name_elem_cls().c_str(), vec.size());
	TRTC_Copy(vec, cpy);

	Functor src_scan({ {"stencil", &stencil}, { "pred", &pred } }, { "idx" },
		"        return pred(stencil[idx]) ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");
	DVVector inds("Pair<uint32_t, uint32_t>", vec.size());
	Functor plus({}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");
	if (!general_scan(vec.size(), src_scan, inds, plus)) return (uint32_t)(-1);
	Pair<uint32_t, uint32_t> ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + vec.size() - 1), sizeof(Pair<uint32_t, uint32_t>));

	static TRTC_For s_for_scatter({ "vec_cpy", "inds", "vec", "count" }, "idx",
		"    if ((idx==0 && inds[idx].first>0) || (idx>0 && inds[idx].first>inds[idx-1].first))\n"
		"        vec[inds[idx].first -1] = vec_cpy[idx];\n"
		"    else\n"
		"        vec[count + inds[idx].second - 1] = vec_cpy[idx];\n"
	);
	DVUInt32 dvcount(ret.first);
	const DeviceViewable* args[] = { &cpy, &inds, &vec, &dvcount };
	if (!s_for_scatter.launch_n(vec.size(), args)) return (uint32_t)(-1);
	return ret.first;
}

uint32_t TRTC_Partition_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_true, DVVectorLike& vec_false, const Functor& pred)
{
	Functor src_scan({ {"src", &vec_in},  { "pred", &pred } }, { "idx" },
		"        return pred(src[idx]) ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");
	DVVector inds("Pair<uint32_t, uint32_t>", vec_in.size());
	Functor plus({}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");
	if (!general_scan(vec_in.size(), src_scan, inds, plus)) return (uint32_t)(-1);
	Pair<uint32_t, uint32_t> ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + vec_in.size() - 1), sizeof(Pair<uint32_t, uint32_t>));

	static TRTC_For s_for_scatter({ "vec_in", "inds", "vec_true", "vec_false" }, "idx",
		"    if ((idx==0 && inds[idx].first>0) || (idx>0 && inds[idx].first>inds[idx-1].first))\n"
		"        vec_true[inds[idx].first -1] = vec_in[idx];\n"
		"    else\n"
		"        vec_false[inds[idx].second - 1] = vec_in[idx];\n"
	);

	const DeviceViewable* args[] = { &vec_in, &inds, &vec_true, &vec_false };
	if (!s_for_scatter.launch_n(vec_in.size(), args)) return (uint32_t)(-1);
	return ret.first;
}

uint32_t TRTC_Partition_Copy_Stencil(const DVVectorLike& vec_in, const DVVectorLike& stencil, DVVectorLike& vec_true, DVVectorLike& vec_false, const Functor& pred)
{
	Functor src_scan({ {"stencil", &stencil}, { "pred", &pred } }, { "idx" },
		"        return pred(stencil[idx]) ? Pair<uint32_t, uint32_t>({(uint32_t)1, (uint32_t)0}):Pair<uint32_t, uint32_t>({(uint32_t)0, (uint32_t)1});\n");
	DVVector inds("Pair<uint32_t, uint32_t>", vec_in.size());
	Functor plus({}, { "x", "y" },
		"        return Pair<uint32_t, uint32_t>({x.first + y.first , x.second + y.second });\n");
	if (!general_scan(vec_in.size(), src_scan, inds, plus)) return (uint32_t)(-1);
	Pair<uint32_t, uint32_t> ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((Pair<uint32_t, uint32_t>*)inds.data() + vec_in.size() - 1), sizeof(Pair<uint32_t, uint32_t>));
	
	static TRTC_For s_for_scatter({ "vec_in", "inds", "vec_true", "vec_false"}, "idx",
		"    if ((idx==0 && inds[idx].first>0) || (idx>0 && inds[idx].first>inds[idx-1].first))\n"
		"        vec_true[inds[idx].first -1] = vec_in[idx];\n"
		"    else\n"
		"        vec_false[inds[idx].second - 1] = vec_in[idx];\n"
	);

	const DeviceViewable* args[] = { &vec_in, &inds, &vec_true, &vec_false};
	if (!s_for_scatter.launch_n(vec_in.size(), args)) return (uint32_t)(-1);
	return ret.first;
}

bool TRTC_Partition_Point(const DVVectorLike& vec, const Functor& pred, size_t& result)
{
	if (vec.size() == 0) return false;

	static TRTC_Kernel s_kernel(
		{ "num_grps", "vec", "begin", "pred", "range_out", "size_grp", "div_id" },
		"    size_t id = threadIdx.x+blockIdx.x*blockDim.x;\n"
		"    if (id>=num_grps) return;"
		"    size_t begin_grp = size_grp*id + begin;\n"
		"    size_t end_grp = begin_grp + size_grp;\n"
		"    if (id>=div_id)\n"
		"    {\n"
		"        begin_grp += id - div_id;\n"
		"        end_grp = begin_grp + size_grp + 1;\n"
		"    }\n"
		"    if ( (id==0 || pred(vec[begin_grp-1])) && !pred(vec[end_grp-1]) )\n"
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
		const DeviceViewable* _args[] = { &_dv_num_grps, &vec, &_dv_begin, &pred, &dv_range_out, &_dv_size_grp, &_dv_div_id };
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
		cuMemcpyHtoD((CUdeviceptr)dv_range_out.data(), h_range_out, sizeof(size_t) * 2);

		const DeviceViewable* args[] = { &dv_num_grps, &vec, &dv_begin, &pred, &dv_range_out, &dv_size_grp, &dv_div_id };
		if (!s_kernel.launch({ (unsigned)numBlocks, 1,1 }, { 128, 1, 1 }, args)) return false;
		dv_range_out.to_host(h_range_out);
		s_begin = h_range_out[0];
		s_end = h_range_out[1];
	} while (s_end > 0 && s_end > s_begin + 1);

	result = s_begin;
	return true;
}

bool TRTC_Is_Partitioned(const DVVectorLike& vec, const Functor& pred, bool& result)
{
	if (vec.size() == 0)
	{
		result = true;
		return true;
	}
	static TRTC_For s_for({ "vec", "pred", "res" }, "idx",
		"    if (!pred(vec[idx]) && pred(vec[idx+1])) res[0] = false;\n");

	result = true;
	DVVector dvres("bool", 1, &result);
	const DeviceViewable* args[] = { &vec, &pred, &dvres };
	if (!s_for.launch_n(vec.size()-1, args)) return false;
	dvres.to_host(&result);
	return true;
}

