#include <memory>
#include <vector>
#include "general_scan_by_key.h"
#include "fill.h"

#define BLOCK_SIZE 256

static bool s_scan_block(size_t n, const Functor& value_in,
	const DVVectorLike& key_in, DVVectorLike& active, DVVectorLike& value_out, 
	DVVectorLike& key_out_b, DVVectorLike& value_out_b, DVVectorLike& active_out_b,
	const Functor& binary_pred, const Functor& binary_op)
{
	static TRTC_Kernel s_kernel({ "value_in", "n", "key_in", "active", "value_out", "key_out_b",
	"value_out_b", "active_out_b", "binary_pred", "binary_op" },
		"    extern __shared__ unsigned char s_buf[];\n"
		"    decltype(key_in)::value_t* s_keys = (decltype(key_in)::value_t*)s_buf;\n"
		"    decltype(value_out)::value_t* s_values =  (decltype(value_out)::value_t*)(s_keys + blockDim.x*2);\n"
		"    bool* s_active = (bool*)(s_values + blockDim.x*2);\n"
		"    unsigned i = threadIdx.x + blockIdx.x*blockDim.x*2;\n"
		"    if (i<n)\n"
		"    {\n"
		"        s_keys[threadIdx.x] = key_in[i];\n"
		"        s_values[threadIdx.x] = (decltype(value_out)::value_t)value_in(i);\n"
		"        s_active[threadIdx.x] = active[i];\n"
		"    }\n"
		"    i = threadIdx.x + blockDim.x + blockIdx.x*blockDim.x*2;\n"
		"    if (i<n)\n"
		"    {\n"
		"        s_keys[threadIdx.x + blockDim.x] = key_in[i];\n"
		"        s_values[threadIdx.x + blockDim.x] = (decltype(value_out)::value_t)value_in(i);\n"
		"        s_active[threadIdx.x + blockDim.x] = active[i];\n"
		"    }\n"
		"    __syncthreads();\n"
		"    unsigned half_size_group = 1;\n"
		"    unsigned size_group = 2;\n"
		"    while(half_size_group <= blockDim.x)\n"
		"    {\n"
		"        unsigned gid = threadIdx.x / half_size_group;\n"
		"        unsigned tid = gid*size_group + half_size_group + threadIdx.x % half_size_group;\n "
		"        i = tid + blockIdx.x*blockDim.x*2;\n"
		"        if (i < n)\n"
		"        {\n"
		"            unsigned end_last = gid*size_group + half_size_group -1;\n"
		"            bool active = false;\n"
		"            if (s_active[tid] && binary_pred(s_keys[end_last],s_keys[tid]))\n"
		"            {\n"
		"                s_values[tid] = binary_op(s_values[end_last], s_values[tid]);\n"
		"                active = s_active[end_last];\n"
		"            }\n"
		"            s_active[tid] = active;\n"
		"        }\n"
		"        half_size_group = half_size_group << 1;"
		"        size_group = size_group << 1;"
		"        __syncthreads();\n"
		"    }\n"
		"    i = threadIdx.x + blockIdx.x*blockDim.x*2;\n"
		"    if (i<n)\n"
		"    {\n"
		"        value_out[i]= s_values[threadIdx.x];\n"
		"        active[i] = s_active[threadIdx.x];\n"
		"    }\n"
		"    i = threadIdx.x + blockDim.x + blockIdx.x*blockDim.x*2;\n"
		"    if (i<n)\n"
		"    {\n"
		"        value_out[i]= s_values[threadIdx.x + blockDim.x];\n"
		"        active[i] = s_active[threadIdx.x + blockDim.x];\n"
		"    }\n"
		"    if (threadIdx.x == 0)\n"
		"    {\n"
		"        unsigned tid = blockDim.x*2 - 1;"
		"        i = tid + blockIdx.x*blockDim.x*2;\n"
		"        if (i >= n) tid = n - 1 - blockIdx.x*blockDim.x*2;\n"
		"        key_out_b[blockIdx.x] = s_keys[tid];"
		"        value_out_b[blockIdx.x] = s_values[tid];"
		"        active_out_b[blockIdx.x] = s_active[tid];"
		"    }\n");
	unsigned blocks = (unsigned)((n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
	unsigned size_shared = (unsigned)((key_in.elem_size() + value_out.elem_size() + sizeof(bool))*BLOCK_SIZE * 2);
	DVSizeT dv_n(n);
	const DeviceViewable* args[] = { &value_in, &dv_n, &key_in,  &active, &value_out, 
		&key_out_b, &value_out_b, &active_out_b, &binary_pred, &binary_op };
	return s_kernel.launch({ blocks,1,1 }, { BLOCK_SIZE ,1,1 }, args, size_shared);
}

static bool s_additional(const DVVectorLike& key, DVVectorLike& value, DVVectorLike& active,
	const DVVectorLike& key_b, const DVVector& value_b, const DVVector& active_b, 
	const Functor& binary_pred, const Functor& binary_op, size_t n)
{
	static TRTC_Kernel s_kernel({ "key", "value", "active", "key_b", "value_b", "active_b",
		"binary_pred", "binary_op", "n" },
		"    unsigned i = threadIdx.x + blockIdx.x*blockDim.x+ blockDim.x*2;\n"
		"    if (i >= n) return;\n"
		"    bool set_active;\n"
		"    if (active[i] && binary_pred(key_b[blockIdx.x/2], key[i]))\n"
		"    {\n"
		"        value[i] = binary_op(value_b[blockIdx.x/2], value[i]);\n"
		"        set_active = active_b[blockIdx.x/2];\n"
		"    }\n"
		"    active[i] = set_active;");

	unsigned blocks = (unsigned)((n - BLOCK_SIZE - 1) / BLOCK_SIZE);
	DVSizeT dv_n(n);
	const DeviceViewable* args[] = { &key, &value, &active, &key_b, &value_b, &active_b, &binary_pred, &binary_op, &dv_n };
	return s_kernel.launch({ blocks,1,1 }, { BLOCK_SIZE ,1,1 }, args);
}


bool general_scan_by_key(size_t n, const Functor& value_in, const DVVectorLike& key, DVVectorLike& value_out, const Functor& binary_pred, const Functor& binary_op)
{
	DVVector dvactive("bool", n);
	TRTC_Fill(dvactive, DVBool(true));
	std::shared_ptr<DVVector> p_key_out_b(new DVVector(key.name_elem_cls().c_str(), (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));
	std::shared_ptr<DVVector> p_value_out_b(new DVVector(value_out.name_elem_cls().c_str(), (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));
	std::shared_ptr<DVVector> p_active_out_b(new DVVector("bool", (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));
	if (!s_scan_block(n, value_in, key, dvactive, value_out, *p_key_out_b, *p_value_out_b, *p_active_out_b, binary_pred, binary_op)) return false;
	std::vector<std::shared_ptr<DVVector>> bufs_key;
	std::vector<std::shared_ptr<DVVector>> bufs_value;
	std::vector<std::shared_ptr<DVVector>> bufs_active;
	while (p_key_out_b->size() > 1)
	{
		bufs_key.push_back(p_key_out_b);
		bufs_value.push_back(p_value_out_b);
		bufs_active.push_back(p_active_out_b);
		DVVector* p_in_key = &*p_key_out_b;
		DVVector* p_in_value = &*p_value_out_b;
		DVVector* p_in_active = &*p_active_out_b;
		size_t n2 = p_in_key->size();
		p_key_out_b = std::shared_ptr<DVVector>(new DVVector(key.name_elem_cls().c_str(), (n2 + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));
		p_value_out_b = std::shared_ptr<DVVector>(new DVVector(value_out.name_elem_cls().c_str(), (n2 + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));
		p_active_out_b = std::shared_ptr<DVVector>(new DVVector("bool", (n2 + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));
		Functor src2({ {"vec", p_in_value} }, { "idx" }, "        return vec[idx];\n");
		if (!s_scan_block(n2, src2, *p_in_key, *p_in_active, *p_in_value, *p_key_out_b, *p_value_out_b, *p_active_out_b, binary_pred, binary_op)) return false;
	}

	for (int i = (int)bufs_key.size() - 2; i >= 0; i--)
		if (!s_additional(*bufs_key[i], *bufs_value[i], *bufs_active[i],
			*bufs_key[i+1], *bufs_value[i+1], *bufs_active[i+1], binary_pred, binary_op, bufs_key[i]->size())) return false;

	if (bufs_key.size() > 0)
		if (!s_additional(key, value_out, dvactive, *bufs_key[0], *bufs_value[0], *bufs_active[0], binary_pred, binary_op, n)) return false;

	return true;
}