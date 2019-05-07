#include "cuda_wrapper.h"
#include <memory>
#include <memory.h>
#include "general_reduce.h"

#define BLOCK_SIZE 256

bool s_reduce(TRTCContext& ctx, const DVVector& src, DVVector& res, const Functor& binary_op)
{
	static TRTC_Kernel s_kernel({ "view_src", "view_res", "n", "binary_op" },
		"    extern __shared__ decltype(view_src)::value_t s_buf[];\n"
		"    unsigned tid = threadIdx.x;\n"
		"    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;\n"
		"    decltype(view_src)::value_t& here=s_buf[tid];\n"
		"    if (i<n)\n"
		"        here = view_src[i];\n"
		"    __syncthreads();\n"
		"    for (unsigned s = blockDim.x/2; s>0; s>>=1)\n    {\n"
		"        if (tid < s && i+s<n)\n"
		"            here = (decltype(view_src)::value_t) binary_op(here, s_buf[tid + s]);\n"
		"        __syncthreads();\n    }\n"
		"    if (tid==0) view_res[blockIdx.x] = here;"
	);

	unsigned n = (unsigned) src.size();
	unsigned blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned size_shared = (unsigned)(src.elem_size()*BLOCK_SIZE);
	DVUInt32 dv_n(n);
	const DeviceViewable* args[] = { &src, &res, &dv_n, &binary_op};
	return s_kernel.launch(ctx, { blocks,1,1 }, { BLOCK_SIZE ,1,1 }, args, size_shared);
}

bool general_reduce(TRTCContext& ctx, size_t n, const char* name_cls, const Functor& src, const Functor& binary_op, ViewBuf& ret_buf)
{
	if (n < 1) return false;

	unsigned blocks = (unsigned)(n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	std::shared_ptr<DVVector> res(new DVVector(ctx, name_cls, blocks));

	// first round
	{
		static TRTC_Kernel s_kernel({ "view_res", "n", "src", "binary_op" },
			"    extern __shared__ decltype(view_res)::value_t s_buf[];\n"
			"    unsigned tid = threadIdx.x;\n"
			"    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;\n"
			"    decltype(view_res)::value_t& here=s_buf[tid];\n"
			"    if (i<n) here=src(i);\n"
			"    __syncthreads();\n"
			"    for (unsigned s = blockDim.x/2; s>0; s>>=1)\n    {\n"
			"        if (tid < s && i+s<n)\n"
			"            here = binary_op(here, s_buf[tid + s]);\n"
			"        __syncthreads();\n    }\n"
			"    if (tid==0) view_res[blockIdx.x] = here;");

		DVUInt32 dv_n((unsigned)n);
		unsigned size_shared = (unsigned)(res->elem_size()*BLOCK_SIZE);
		const DeviceViewable* args[] = { &*res, &dv_n, &src, &binary_op };
		if (!s_kernel.launch(ctx, { blocks,1,1 }, { BLOCK_SIZE ,1,1 }, args, size_shared)) return false;
	}

	while (res->size() > 1)
	{
		std::shared_ptr<DVVector> src = res;
		n = (unsigned)src->size();
		blocks = (unsigned)(n + BLOCK_SIZE - 1) / BLOCK_SIZE;		
		res = std::shared_ptr<DVVector>(new DVVector(ctx, name_cls, blocks));
		if (!s_reduce(ctx, *src, *res, binary_op)) return false;
	}

	ret_buf.resize(res->elem_size());
	cuMemcpyDtoH(ret_buf.data(), (CUdeviceptr)res->data(), res->elem_size());
	return true;
}