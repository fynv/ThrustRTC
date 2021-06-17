#include "cuda_wrapper.h"
#include "general_scan.h"

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

uint32_t general_copy_if(size_t n, const Functor& src_scan, const DVVectorLike& vec_in, DVVectorLike& vec_out)
{
	DVVector inds("uint32_t", n);
	Functor plus("Plus");
	if (!general_scan(n, src_scan, inds, plus)) return (uint32_t)(-1);

	uint32_t ret;
	if (!CheckCUresult(cuMemcpyDtoH(&ret, (CUdeviceptr)((uint32_t*)inds.data() + n - 1), sizeof(uint32_t)), " cuMemcpyDtoH()")) return (uint32_t)(-1);

	static TRTC_For s_for_scatter({ "vec_in", "inds", "vec_out"}, "idx",
		"    if ((idx==0 && inds[idx]>0) || (idx>0 && inds[idx]>inds[idx-1])) vec_out[inds[idx]-1]=vec_in[idx];\n"
	);

	const DeviceViewable* args[] = { &vec_in, &inds, &vec_out };
	if (!s_for_scatter.launch_n(n, args)) return (uint32_t)(-1);
	return ret;
}

uint32_t general_copy_if(size_t n, const Functor& src_scan, const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, DVVectorLike& vec_out1, DVVectorLike& vec_out2)
{
	DVVector inds("uint32_t", n);
	Functor plus("Plus");
	if (!general_scan(n, src_scan, inds, plus)) return (uint32_t)(-1);

	uint32_t ret;
	if (!CheckCUresult(cuMemcpyDtoH(&ret, (CUdeviceptr)((uint32_t*)inds.data() + n - 1), sizeof(uint32_t)), " cuMemcpyDtoH()")) return (uint32_t)(-1);

	static TRTC_For s_for_scatter({ "vec_in1", "vec_in2", "inds", "vec_out1", "vec_out2" }, "idx",
		"    if ((idx==0 && inds[idx]>0) || (idx>0 && inds[idx]>inds[idx-1]))\n"
		"    {\n"
		"        vec_out1[inds[idx]-1]=vec_in1[idx];\n"
		"        vec_out2[inds[idx]-1]=vec_in2[idx];\n"
		"    }\n"
	);

	const DeviceViewable* args[] = { &vec_in1, &vec_in2, &inds, &vec_out1, &vec_out2 };
	if (!s_for_scatter.launch_n(n, args)) return (uint32_t)(-1);
	return ret;
}
