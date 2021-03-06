#include "merge_sort.h"
#include "copy.h"
#include "fake_vectors/DVRange.h"

#define BLOCK_SIZE 256

bool merge_sort(DVVectorLike& vec, const Functor& comp)
{
	// block level
	{
		static TRTC_Kernel s_ker_block(
			{ "vec", "comp" },
			"    size_t end = vec.size();"
			"    extern __shared__ decltype(vec)::value_t s_buf[];\n"
			"    unsigned i = threadIdx.x + blockIdx.x*blockDim.x*2;\n"
			"    if (i<end) s_buf[threadIdx.x]= vec[i];\n"
			"    i = threadIdx.x + blockDim.x + blockIdx.x*blockDim.x*2;\n"
			"    if (i<end) s_buf[threadIdx.x + blockDim.x] = vec[i];\n"
			"    __syncthreads();\n"
			"    unsigned half_size_group = 1;\n"
			"    unsigned size_group = 2;\n"
			"    while(half_size_group <= blockDim.x)\n"
			"    {\n"
			"        unsigned gid = threadIdx.x / half_size_group;\n"
			"        decltype(vec)::value_t v1, v2;\n"
			"        unsigned pos1= (unsigned)(-1);\n"
			"        unsigned pos2= (unsigned)(-1);\n"
			"        do"
			"        {\n"
			"            i = gid*size_group + half_size_group + blockIdx.x*blockDim.x*2;\n"
			"            if (i>=end) break;"
			"            unsigned tid = gid*size_group + threadIdx.x % half_size_group;\n "
			"            v1 = s_buf[tid];\n"
			"            decltype(vec)::value_t* p_search = s_buf +  gid*size_group + half_size_group;\n"
			"            i =  (gid+1)*size_group + blockIdx.x*blockDim.x*2;\n"
			"            unsigned n = i<end ? half_size_group : end - blockIdx.x*blockDim.x*2 - gid*size_group - half_size_group;\n"
			"            pos1 = threadIdx.x % half_size_group + d_lower_bound_s(p_search, n, v1, comp);\n"
			"        } while(false);\n"
			"        do"
			"        {\n"
			"            unsigned tid = gid*size_group + half_size_group + threadIdx.x % half_size_group;\n"
			"            i = tid + blockIdx.x*blockDim.x*2;\n"
			"            if (i>=end) break;"
			"            v2 = s_buf[tid];\n"
			"            decltype(vec)::value_t* p_search = s_buf +  gid*size_group;\n"
			"            pos2 = threadIdx.x % half_size_group + d_upper_bound_s(p_search, half_size_group, v2, comp);\n"
			"        } while(false);\n"
			"        __syncthreads();\n"
			"        if (pos1!=(unsigned)(-1)) s_buf[gid*size_group + pos1]=v1;\n"
			"        if (pos2!=(unsigned)(-1)) s_buf[gid*size_group + pos2]=v2;\n"
			"        __syncthreads();\n"
			"        half_size_group = half_size_group << 1;"
			"        size_group = size_group << 1;"
			"    }\n"
			"    i = threadIdx.x + blockIdx.x*blockDim.x*2;\n"
			"    if (i < end) vec[i] = s_buf[threadIdx.x];\n"
			"    i = threadIdx.x + blockDim.x + blockIdx.x*blockDim.x*2;\n"
			"    if (i < end) vec[i] = s_buf[threadIdx.x + blockDim.x];\n"
		);

		unsigned blocks = (unsigned)((vec.size() + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
		unsigned size_shared = (unsigned)(vec.elem_size()*BLOCK_SIZE * 2);
		const DeviceViewable* args[] = { &vec, &comp };
		if (!s_ker_block.launch({ blocks, 1, 1 }, { BLOCK_SIZE, 1, 1 }, args, size_shared)) return false;
	}

	// global level
	size_t half_size_group = BLOCK_SIZE * 2;
	size_t size_group = half_size_group * 2;
	if (vec.size() > half_size_group)
	{
		size_t size = vec.size();
		DVVector tmp1(vec.name_elem_cls().c_str(), size);
		DVVector tmp2(vec.name_elem_cls().c_str(), size);
		if (!TRTC_Copy(vec, tmp1)) return false;
		DVVector* p1 = &tmp1;
		DVVector* p2 = &tmp2;
		DVSizeT dvsize(size);
		while (size > half_size_group)
		{
			static TRTC_Kernel s_ker_global(
				{ "vec_in", "vec_out", "size", "comp", "half_size_group" },
				"    size_t idx = threadIdx.x + blockIdx.x*blockDim.x;\n"
				"    size_t size_group = half_size_group*2;\n"
				"    size_t gid = idx / half_size_group;\n "
				// left half
				"    size_t pos_in = gid*size_group + idx % half_size_group;\n"
				"    size_t begin = gid*size_group + half_size_group;\n"
				"    size_t end = (gid+1)*size_group;\n"
				"    if (end>size) end = size;\n"
				"    size_t pos_out = pos_in + d_lower_bound(vec_in, vec_in[pos_in], comp, begin, end) - begin;\n"
				"    vec_out[pos_out] = vec_in[pos_in];\n"
				// right half
				"    pos_in = gid*size_group + half_size_group + idx % half_size_group;\n"
				"    if (pos_in>=size) return;\n"
				"    begin = gid*size_group;\n"
				"    end =  begin + half_size_group;\n"
				"    pos_out = pos_in - half_size_group + d_upper_bound(vec_in, vec_in[pos_in], comp, begin, end) - begin;\n"
				"    vec_out[pos_out] = vec_in[pos_in];\n"
			);

			unsigned blocks = (unsigned)((size + half_size_group - 1) / half_size_group / 2 * (half_size_group / BLOCK_SIZE));
			DVSizeT dv_half_size_group(half_size_group);
			const DeviceViewable* args[] = { p1, p2, &dvsize, &comp, &dv_half_size_group };
			if (!s_ker_global.launch({ blocks, 1, 1 }, { BLOCK_SIZE, 1, 1 }, args)) return false;

			half_size_group = half_size_group << 1;
			size_group = size_group << 1;
			{
				DVVector* tmp = p1;
				p1 = p2;
				p2 = tmp;
			}
		}
		if (!TRTC_Copy(*p1, vec)) return false;
	}
	return true;
}

bool merge_sort_by_key(DVVectorLike& keys, DVVectorLike& values, const Functor& comp)
{
	// block level
	{
		static TRTC_Kernel s_ker_block(
			{ "keys", "values", "comp" },
			"    size_t end_keys = keys.size();"
			"    extern __shared__ unsigned char s_buf[];\n"
			"    decltype(keys)::value_t* s_keys = (decltype(keys)::value_t*)s_buf;\n"
			"    decltype(values)::value_t* s_values = (decltype(values)::value_t*)(s_keys + blockDim.x*2);\n"
			"    unsigned i = threadIdx.x + blockIdx.x*blockDim.x*2;\n"
			"    if (i<end_keys)\n"
			"    {\n"
			"        s_keys[threadIdx.x] = keys[i];\n"
			"        s_values[threadIdx.x] = values[i];\n"
			"    }\n"
			"    i = threadIdx.x + blockDim.x + blockIdx.x*blockDim.x*2;\n"
			"    if (i<end_keys)\n"
			"    {\n"
			"        s_keys[threadIdx.x + blockDim.x] = keys[i];\n"
			"        s_values[threadIdx.x + blockDim.x] = values[i];\n"
			"    }\n"
			"    __syncthreads();\n"
			"    unsigned half_size_group = 1;\n"
			"    unsigned size_group = 2;\n"
			"    while(half_size_group <= blockDim.x)\n"
			"    {\n"
			"        unsigned gid = threadIdx.x / half_size_group;\n"
			"        decltype(keys)::value_t key1, key2;\n"
			"        decltype(values)::value_t v1, v2;\n"
			"        unsigned pos1= (unsigned)(-1);\n"
			"        unsigned pos2= (unsigned)(-1);\n"
			"        do"
			"        {\n"
			"            i = gid * size_group + half_size_group + blockIdx.x*blockDim.x * 2; \n"
			"            if (i>=end_keys) break;\n"
			"            unsigned tid = gid*size_group + threadIdx.x % half_size_group;\n "
			"            key1 = s_keys[tid];\n"
			"            v1 = s_values[tid];\n"
			"            decltype(keys)::value_t* p_search = s_keys +  gid*size_group + half_size_group;\n"
			"            i =  (gid+1)*size_group + blockIdx.x*blockDim.x*2;\n"
			"            unsigned n = i<end_keys ? half_size_group : end_keys - blockIdx.x*blockDim.x*2 - gid*size_group - half_size_group;\n"
			"            pos1 = threadIdx.x % half_size_group + d_lower_bound_s(p_search, n, key1, comp);\n"
			"        } while(false);\n"
			"        do"
			"        {\n"
			"            unsigned tid = gid*size_group + half_size_group + threadIdx.x % half_size_group;\n"
			"            i = tid + blockIdx.x*blockDim.x*2;\n"
			"            if (i>=end_keys) break;\n"
			"            key2 = s_keys[tid];\n"
			"            v2 = s_values[tid];\n"
			"            decltype(keys)::value_t* p_search = s_keys +  gid*size_group;\n"
			"            pos2 = threadIdx.x % half_size_group + d_upper_bound_s(p_search, half_size_group, key2, comp);\n"
			"        } while(false);\n"
			"        __syncthreads();\n"
			"        if (pos1!=(unsigned)(-1))\n"
			"        {\n"
			"            s_keys[gid*size_group + pos1]=key1;\n"
			"            s_values[gid*size_group + pos1]=v1;\n"
			"        }\n"
			"        if (pos2!=(unsigned)(-1))\n"
			"        {\n"
			"            s_keys[gid*size_group + pos2]=key2;\n"
			"            s_values[gid*size_group + pos2]=v2;\n"
			"        }\n"
			"        __syncthreads();\n"
			"        half_size_group = half_size_group << 1;"
			"        size_group = size_group << 1;"
			"    }\n"
			"    i = threadIdx.x + blockIdx.x*blockDim.x*2;\n"
			"    if (i<end_keys)\n"
			"    {\n"
			"        keys[i]= s_keys[threadIdx.x];\n"
			"        values[i] = s_values[threadIdx.x];\n"
			"    }\n"
			"    i = threadIdx.x + blockDim.x + blockIdx.x*blockDim.x*2;\n"
			"    if (i<end_keys)\n"
			"    {\n"
			"        keys[i] = s_keys[threadIdx.x + blockDim.x];\n"
			"        values[i] = s_values[threadIdx.x + blockDim.x];\n"
			"    }\n"
		);

		unsigned blocks = (unsigned)((keys.size() + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
		unsigned size_shared = (unsigned)((keys.elem_size() + values.elem_size())*BLOCK_SIZE * 2);
		const DeviceViewable* args[] = { &keys, &values, &comp };
		if (!s_ker_block.launch({ blocks, 1, 1 }, { BLOCK_SIZE, 1, 1 }, args, size_shared)) return false;
	}

	// global level
	size_t half_size_group = BLOCK_SIZE * 2;
	size_t size_group = half_size_group * 2;
	if (keys.size() > half_size_group)
	{
		size_t size_keys = keys.size();
		size_t size_values = keys.size();
		DVVector tmp_keys1(keys.name_elem_cls().c_str(), size_keys);
		DVVector tmp_keys2(keys.name_elem_cls().c_str(), size_keys);
		DVVector tmp_values1(values.name_elem_cls().c_str(), size_keys);
		DVVector tmp_values2(values.name_elem_cls().c_str(), size_keys);
		if (!TRTC_Copy(keys, tmp_keys1)) return false;
		if (size_values == size_keys)
		{
			if (!TRTC_Copy(values, tmp_values1)) return false;
		}
		else
		{
			if (!TRTC_Copy(DVRange(values, 0, size_keys), tmp_values1)) return false;
		}

		DVVector* pkeys1 = &tmp_keys1;
		DVVector* pkeys2 = &tmp_keys2;
		DVVector* pvalues1 = &tmp_values1;
		DVVector* pvalues2 = &tmp_values2;
		DVSizeT dvsize(size_keys);
		while (size_keys > half_size_group)
		{
			static TRTC_Kernel s_ker_global(
				{ "keys_in", "keys_out", "values_in", "values_out", "size", "comp", "half_size_group" },
				"    size_t idx = threadIdx.x + blockIdx.x*blockDim.x;\n"
				"    size_t size_group = half_size_group*2;\n"
				"    size_t gid = idx / half_size_group;\n "
				// left half
				"    size_t pos_in = gid*size_group + idx % half_size_group;\n"
				"    size_t begin = gid*size_group + half_size_group;\n"
				"    size_t end = (gid+1)*size_group;\n"
				"    if (end>size) end = size;\n"
				"    size_t pos_out = pos_in + d_lower_bound(keys_in, keys_in[pos_in], comp, begin, end) - begin;\n"
				"    keys_out[pos_out] = keys_in[pos_in];\n"
				"    values_out[pos_out] = values_in[pos_in];\n"
				// right half
				"    pos_in = gid*size_group + half_size_group + idx % half_size_group;\n"
				"    if (pos_in>=size) return;\n"
				"    begin = gid*size_group;\n"
				"    end =  begin + half_size_group;\n"
				"    pos_out = pos_in - half_size_group + d_upper_bound(keys_in, keys_in[pos_in], comp, begin, end) - begin;\n"
				"    keys_out[pos_out] = keys_in[pos_in];\n"
				"    values_out[pos_out] = values_in[pos_in];\n"
			);

			unsigned blocks = (unsigned)((size_keys + half_size_group - 1) / half_size_group / 2 * (half_size_group / BLOCK_SIZE));
			DVSizeT dv_half_size_group(half_size_group);
			const DeviceViewable* args[] = { pkeys1, pkeys2, pvalues1, pvalues2, &dvsize, &comp, &dv_half_size_group };
			if (!s_ker_global.launch({ blocks, 1, 1 }, { BLOCK_SIZE, 1, 1 }, args)) return false;
			half_size_group = half_size_group << 1;
			size_group = size_group << 1;
			{
				DVVector* tmp = pkeys1;
				pkeys1 = pkeys2;
				pkeys2 = tmp;
				tmp = pvalues1;
				pvalues1 = pvalues2;
				pvalues2 = tmp;
			}
		}
		if (!TRTC_Copy(*pkeys1, keys)) return false;
		if (!TRTC_Copy(*pvalues1, values)) return false;
	}

	return true;
}
