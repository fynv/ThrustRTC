#include <cuda.h>
#include <memory>
#include <memory.h>
#include "general_reduce.h"

#define BLOCK_SIZE 256

bool s_reduce(TRTCContext& ctx, const DVVector& src, DVVector& res, const Functor& binary_op)
{
	unsigned n = (unsigned) src.size();
	unsigned blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	DVUInt32 dv_n(n);
	std::vector<TRTCContext::AssignedParam> arg_map = binary_op.arg_map;
	arg_map.push_back({ "_view_src", &src });
	arg_map.push_back({ "_view_res", &res });
	arg_map.push_back({ "_n", &dv_n });
	unsigned size_shared = (unsigned)(src.elem_size()*BLOCK_SIZE);

	return ctx.launch_kernel({ blocks,1,1 }, { BLOCK_SIZE ,1,1 }, arg_map,
		(std::string("    extern __shared__ ") + src.name_elem_cls() + " _s_buf[];\n"
			"    unsigned _tid = threadIdx.x;\n"
			"    unsigned _i = blockIdx.x*blockDim.x + threadIdx.x;\n"
			"    decltype(_view_src)::value_t& _here=_s_buf[_tid];\n"
			"    if (_i<_n)\n"
			"        _here = _view_src[_i];\n"
			"    __syncthreads();\n"
			"    for (unsigned _s = blockDim.x/2; _s>0; _s>>=1)\n    {\n"
			"        if (_tid < _s && _i+_s<_n)\n        {\n" + binary_op.generate_code("decltype(_view_src)::value_t", { "_here", "_s_buf[_tid + _s]" }) +
			"            _here = " + binary_op.functor_ret + ";\n        }\n"
			"        __syncthreads();\n    }\n"
			"    if (_tid==0) _view_res[blockIdx.x] = _here;"
			).c_str(),
		size_shared);
}

bool general_reduce(TRTCContext& ctx, size_t n, const char* name_cls, const Functor& src, const Functor& binary_op, ViewBuf& ret_buf)
{
	if (n < 1) return false;

	unsigned blocks = (unsigned)(n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	std::shared_ptr<DVVector> res(new DVVector(ctx, name_cls, blocks));

	// first round
	{
		DVUInt32 dv_n((unsigned)n);

		std::vector<TRTCContext::AssignedParam> arg_map(src.arg_map.size() + binary_op.arg_map.size() + 2);
		memcpy(arg_map.data(), src.arg_map.data(), src.arg_map.size() * sizeof(TRTCContext::AssignedParam));
		memcpy(arg_map.data() + src.arg_map.size(), binary_op.arg_map.data(), binary_op.arg_map.size() * sizeof(TRTCContext::AssignedParam));
		TRTCContext::AssignedParam* p_arg_map = &arg_map[src.arg_map.size() + binary_op.arg_map.size()];
		p_arg_map[0] = { "_view_res", &*res };
		p_arg_map[1] = { "_n", &dv_n };
		unsigned size_shared = (unsigned)(res->elem_size()*BLOCK_SIZE);

		if (!ctx.launch_kernel({ blocks,1,1 }, { BLOCK_SIZE ,1,1 }, arg_map,
			(std::string("    extern __shared__ ") + name_cls + " _s_buf[];\n"
				"    unsigned _tid = threadIdx.x;\n"
				"    unsigned _i = blockIdx.x*blockDim.x + threadIdx.x;\n"
				"    " + name_cls + "& _here=_s_buf[_tid];\n"
				"    if (_i<_n)\n    {\n" + src.generate_code(name_cls, { "_i" }) +
				"        _here=" + src.functor_ret + ";\n    }\n"
				"    __syncthreads();\n"
				"    for (unsigned _s = blockDim.x/2; _s>0; _s>>=1)\n    {\n"
				"        if (_tid < _s && _i+_s<_n)\n        {\n" + binary_op.generate_code(name_cls, { "_here", "_s_buf[_tid + _s]" }) +
				"            _here = " + binary_op.functor_ret + ";\n        }\n"
				"        __syncthreads();\n    }\n"
				"    if (_tid==0) _view_res[blockIdx.x] = _here;"
				).c_str(),
			size_shared)) return false;
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