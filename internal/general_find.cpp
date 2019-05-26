#include "general_find.h"

bool general_find(TRTCContext& ctx, size_t begin, size_t end, const Functor src, size_t& result)
{
	static TRTC_Kernel s_kernel(
		{ "src", "result", "begin", "end" },
		"    __shared__ size_t s_result;"
		"    if (threadIdx.x == 0) s_result = (size_t)(-1);\n"
		"    __syncthreads();\n"
		"    size_t id = threadIdx.x+blockIdx.x*blockDim.x + begin;\n"
		"    if (id<end && src(id))\n"
		"       atomicMin(&s_result, id);\n"
		"    __syncthreads();\n"
		"    if (threadIdx.x == 0 && s_result!= (size_t)(-1))\n"
		"        atomicMin(&result[0], s_result);\n"
	);

	DVSizeT _dvbegin(begin);
	DVSizeT _dvend(end);
	result = (size_t)(-1);
	DVVector dvresult(ctx, "size_t", 1, &result);
	const DeviceViewable* _args[] = { &src, &dvresult, &_dvbegin, &_dvend };
	int numBlocks;
	s_kernel.calc_number_blocks(ctx, _args, 128, numBlocks);
	unsigned trunk_size = (unsigned)numBlocks * 128;
	unsigned trunk_begin = (unsigned)begin;
	while (trunk_begin < end)
	{
		unsigned trunk_end = trunk_begin + trunk_size;
		if (trunk_end > end) trunk_end = (unsigned)end;
		DVSizeT dvbegin(trunk_begin);
		DVSizeT dvend(trunk_end);
		const DeviceViewable* args[] = { &src, &dvresult, &dvbegin, &dvend };
		numBlocks = (int)((trunk_end - trunk_begin + 127) / 128);
		if (!s_kernel.launch(ctx, { (unsigned)numBlocks, 1,1 }, { 128, 1, 1 }, args)) return false;
		dvresult.to_host(&result);
		if (result != (size_t)(-1)) break;
		trunk_begin = trunk_end;
	}
	return true;
}
