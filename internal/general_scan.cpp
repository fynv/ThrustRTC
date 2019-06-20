#include <memory.h>
#include <memory>
#include <vector>
#include "general_scan.h"

#define BLOCK_SIZE 256

static bool s_scan_block(size_t n, const Functor& src, DVVectorLike& vec_out, DVVectorLike& vec_out_b, const Functor& binary_op)
{
	static TRTC_Kernel s_kernel({ "vec_out", "vec_out_b", "n", "src", "binary_op" },
		"    extern __shared__ decltype(vec_out)::value_t s_buf[];\n"
		"    unsigned i = threadIdx.x + blockIdx.x*blockDim.x*2;\n"
		"    if (i<n) s_buf[threadIdx.x]= (decltype(vec_out)::value_t)src(i);\n"
		"    i = threadIdx.x + blockDim.x + blockIdx.x*blockDim.x*2;\n"
		"    if (i<n) s_buf[threadIdx.x + blockDim.x] = (decltype(vec_out)::value_t)src(i);\n"
		"    __syncthreads();\n"
		"    unsigned half_size_group = 1;\n"
		"    unsigned size_group = 2;\n"
		"    while(half_size_group <= blockDim.x)\n"
		"    {\n"
		"        unsigned gid = threadIdx.x / half_size_group;\n"
		"        unsigned tid = gid*size_group + half_size_group + threadIdx.x % half_size_group;\n "
		"        i = tid + blockIdx.x*blockDim.x*2;\n"
		"        if (i < n)\n"
		"            s_buf[tid] = binary_op(s_buf[gid*size_group + half_size_group -1], s_buf[tid]);\n"
		"        half_size_group = half_size_group << 1;"
		"        size_group = size_group << 1;"
		"        __syncthreads();\n"
		"    }\n"
		"    i = threadIdx.x + blockIdx.x*blockDim.x*2;\n"
		"    if (i < n) vec_out[i] = s_buf[threadIdx.x];\n"
		"    i = threadIdx.x + blockDim.x + blockIdx.x*blockDim.x*2;\n"
		"    if (i < n) vec_out[i] = s_buf[threadIdx.x + blockDim.x];\n"
		"    if (threadIdx.x == 0)\n"
		"    {\n"
		"        unsigned tid = blockDim.x*2 - 1;\n"
		"        i = tid + blockIdx.x*blockDim.x*2;\n"
		"        if (i >= n) tid = n - 1 - blockIdx.x*blockDim.x*2;\n"
		"        vec_out_b[blockIdx.x] = s_buf[tid];"
		"    }\n");


	unsigned blocks = (unsigned)((n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
	unsigned size_shared = (unsigned)(vec_out.elem_size()*BLOCK_SIZE * 2);
	DVSizeT dv_n(n);
	const DeviceViewable* args[] = { &vec_out, &vec_out_b, &dv_n, &src, &binary_op };
	return s_kernel.launch({ blocks,1,1 }, { BLOCK_SIZE ,1,1 }, args, size_shared);
}

static bool s_additional(DVVectorLike& vec, const DVVectorLike& vec_b, const Functor& binary_op, size_t n)
{
	static TRTC_Kernel s_kernel({ "vec", "vec_b", "begin", "end", "binary_op" },
		"    unsigned i = threadIdx.x + blockIdx.x*blockDim.x;\n"
		"    if (i + begin < end)\n"
		"        vec[i + begin] = binary_op(vec[i + begin], vec_b[blockIdx.x/2]);\n");


	size_t begin = BLOCK_SIZE * 2;
	unsigned blocks = (unsigned)((n - begin + BLOCK_SIZE - 1) / BLOCK_SIZE);
	DVSizeT dvbegin(begin);
	DVSizeT dvend(n);
	const DeviceViewable* args[] = { &vec, &vec_b, &dvbegin, &dvend, &binary_op };
	return s_kernel.launch({ blocks,1,1 }, { BLOCK_SIZE ,1,1 }, args);
}

bool general_scan(size_t n, const Functor& src, DVVectorLike& vec_out, const Functor& binary_op)
{
	std::shared_ptr<DVVector> p_out_b(new DVVector(vec_out.name_elem_cls().c_str(), (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));
	if (!s_scan_block(n, src, vec_out, *p_out_b, binary_op)) return false;
	std::vector<std::shared_ptr<DVVector>> bufs;
	while (p_out_b->size() > 1)
	{
		bufs.push_back(p_out_b);
		DVVector* pbuf = &*p_out_b;
		size_t n2 = p_out_b->size();
		p_out_b = std::shared_ptr<DVVector>(new DVVector(vec_out.name_elem_cls().c_str(), (n2 + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));
		Functor src2({ {"vec", pbuf} }, { "idx" }, "        return vec[idx];\n");
		if (!s_scan_block(n2, src2, *pbuf, *p_out_b, binary_op)) return false;
	}

	for (int i = (int)bufs.size() - 2; i >= 0; i--)
		if (!s_additional(*bufs[i], *bufs[i + 1], binary_op, bufs[i]->size())) return false;

	if (bufs.size() > 0)
		if (!s_additional(vec_out, *bufs[0], binary_op, n)) return false;

	return true;
}

