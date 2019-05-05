#include <memory.h>
#include <memory>
#include <vector>
#include "general_scan.h"

#define BLOCK_SIZE 256

static bool s_scan_block(TRTCContext& ctx, size_t n, const Functor& src, DVVectorLike& vec_out, DVVectorLike& vec_out_b, const Functor& binary_op, size_t begin_out)
{
	unsigned blocks = (unsigned)((n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
	DVSizeT dvbegin_out(begin_out);
	DVSizeT dv_n(n);

	std::vector<TRTCContext::AssignedParam> arg_map(src.arg_map.size() + binary_op.arg_map.size() + 4);
	memcpy(arg_map.data(), src.arg_map.data(), src.arg_map.size() * sizeof(TRTCContext::AssignedParam));
	memcpy(arg_map.data() + src.arg_map.size(), binary_op.arg_map.data(), binary_op.arg_map.size() * sizeof(TRTCContext::AssignedParam));
	TRTCContext::AssignedParam* p_arg_map = &arg_map[src.arg_map.size() + binary_op.arg_map.size()];
	p_arg_map[0] = { "_vec_out", &vec_out };
	p_arg_map[1] = { "_vec_out_b", &vec_out_b };
	p_arg_map[2] = { "_begin_out", &dvbegin_out };
	p_arg_map[3] = { "_n", &dv_n };
	unsigned size_shared = (unsigned)(vec_out.elem_size()*BLOCK_SIZE * 2);

	return ctx.launch_kernel({ blocks,1,1 }, { BLOCK_SIZE ,1,1 }, arg_map,
		(std::string("    extern __shared__ decltype(_vec_out)::value_t _s_buf[];\n") +
			"    unsigned _i = threadIdx.x + blockIdx.x*blockDim.x*2;\n"
			"    if (_i<_n)\n    {\n" + src.generate_code("decltype(_vec_out)::value_t", { "_i" }) +
			"         _s_buf[threadIdx.x]=" + src.functor_ret + ";\n    }\n"
			"    _i = threadIdx.x + blockDim.x + blockIdx.x*blockDim.x*2;\n"
			"    if (_i<_n)\n    {\n" + src.generate_code("decltype(_vec_out)::value_t", { "_i" }) +
			"         _s_buf[threadIdx.x + blockDim.x]=" + src.functor_ret + ";\n    }\n"
			"    __syncthreads();\n"
			"    unsigned _half_size_group = 1;\n"
			"    unsigned _size_group = 2;\n"
			"    while(_half_size_group <= blockDim.x)\n"
			"    {\n"
			"        unsigned _gid = threadIdx.x / _half_size_group;\n"
			"        unsigned _tid = _gid*_size_group + _half_size_group + threadIdx.x % _half_size_group;\n "
			"        _i = _tid + blockIdx.x*blockDim.x*2;\n"
			"        if (_i < _n)\n"
			"        {\n" +
			binary_op.generate_code("decltype(_vec_out)::value_t", { "_s_buf[_gid*_size_group + _half_size_group -1]", "_s_buf[_tid]" }) +
			"            _s_buf[_tid] = " + binary_op.functor_ret + ";\n"
			"        }\n"
			"        _half_size_group = _half_size_group << 1;"
			"        _size_group = _size_group << 1;"
			"        __syncthreads();\n"
			"    }\n"
			"    _i = threadIdx.x + blockIdx.x*blockDim.x*2;\n"
			"    if (_i < _n) _vec_out[_i + _begin_out] = (decltype(_vec_out)::value_t) _s_buf[threadIdx.x];\n"
			"    _i = threadIdx.x + blockDim.x + blockIdx.x*blockDim.x*2;\n"
			"    if (_i < _n) _vec_out[_i + _begin_out] = (decltype(_vec_out)::value_t) _s_buf[threadIdx.x + blockDim.x];\n"
			"    if (threadIdx.x == 0)\n"
			"    {\n"
			"        unsigned _tid = blockDim.x*2 - 1;\n"
			"        _i = _tid + blockIdx.x*blockDim.x*2;\n"
			"        if (_i >= _n) _tid = _n - 1 - blockIdx.x*blockDim.x*2;\n"
			"        _vec_out_b[blockIdx.x] = _s_buf[_tid];"
			"    }\n"
			).c_str(),
		size_shared);
}

static bool s_additional(TRTCContext& ctx, DVVectorLike& vec, const DVVectorLike& vec_b, const Functor& binary_op, size_t begin, size_t end)
{
	begin += BLOCK_SIZE * 2;
	unsigned blocks = (unsigned)((end - begin + BLOCK_SIZE - 1) / BLOCK_SIZE);
	DVSizeT dvbegin(begin);
	DVSizeT dvend(end);

	std::vector<TRTCContext::AssignedParam> arg_map = binary_op.arg_map;
	arg_map.push_back({ "_vec", &vec });
	arg_map.push_back({ "_vec_b", &vec_b });
	arg_map.push_back({ "_begin", &dvbegin });
	arg_map.push_back({ "_end", &dvend });

	return ctx.launch_kernel({ blocks,1,1 }, { BLOCK_SIZE ,1,1 }, arg_map,
		(std::string("    unsigned _i = threadIdx.x + blockIdx.x*blockDim.x;\n") +
			"    if (_i + _begin < _end)\n"
			"    {\n" +
			binary_op.generate_code("decltype(_vec)::value_t", { "_vec[_i + _begin]", "_vec_b[blockIdx.x/2]" }) +
			"        _vec[_i + _begin] = " + binary_op.functor_ret + ";\n"
			"    }\n").c_str());
}

bool general_scan(TRTCContext& ctx, size_t n, const Functor& src, DVVectorLike& vec_out, const Functor& binary_op, size_t begin_out)
{
	std::shared_ptr<DVVector> p_out_b(new DVVector(ctx, vec_out.name_elem_cls().c_str(), (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));
	if (!s_scan_block(ctx, n, src, vec_out, *p_out_b, binary_op, begin_out)) return false;
	std::vector<std::shared_ptr<DVVector>> bufs;
	while (p_out_b->size() > 1)
	{
		bufs.push_back(p_out_b);
		DVVector* pbuf = &*p_out_b;
		size_t n2 = p_out_b->size();
		p_out_b = std::shared_ptr<DVVector>(new DVVector(ctx, vec_out.name_elem_cls().c_str(), (n2 + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));

		Functor src2 = { { {"_vec", pbuf}} , { "_idx" }, "_ret",
		"        _ret = _vec[_idx];\n" };
		if (!s_scan_block(ctx, n2, src2, *pbuf, *p_out_b, binary_op, 0)) return false;
	}

	for (int i = (int)bufs.size() - 2; i >= 0; i--)
		if (!s_additional(ctx, *bufs[i], *bufs[i + 1], binary_op, 0, bufs[i]->size())) return false;

	if (bufs.size() > 0)
		if (!s_additional(ctx, vec_out, *bufs[0], binary_op, begin_out, begin_out + n)) return false;

	return true;
}

