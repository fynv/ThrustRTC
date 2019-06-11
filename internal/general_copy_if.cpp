#include "cuda_wrapper.h"
#include "general_scan.h"

uint32_t general_copy_if(size_t n, const Functor& src_scan, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in, size_t begin_out)
{
	DVVector inds("uint32_t", n);
	Functor plus("Plus");
	if (!general_scan(n, src_scan, inds, plus, 0)) return (uint32_t)(-1);

	uint32_t ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((uint32_t*)inds.data() + n - 1), sizeof(uint32_t));

	static TRTC_For s_for_scatter({ "vec_in", "inds", "vec_out", "begin_in", "begin_out" }, "idx",
		"    if ((idx==0 && inds[idx]>0) || (idx>0 && inds[idx]>inds[idx-1])) vec_out[inds[idx]-1 + begin_out]=vec_in[idx +begin_in];\n"
	);

	DVSizeT dvbegin_in(begin_in);
	DVSizeT dvbegin_out(begin_out);
	const DeviceViewable* args[] = { &vec_in, &inds, &vec_out, &dvbegin_in, &dvbegin_out };
	if (!s_for_scatter.launch_n(n, args)) return (uint32_t)(-1);
	return ret;
}

uint32_t general_copy_if(size_t n, const Functor& src_scan, const DVVectorLike& vec_in1, const DVVectorLike& vec_in2,
	DVVectorLike& vec_out1, DVVectorLike& vec_out2, size_t begin_in1, size_t begin_in2, size_t begin_out1, size_t begin_out2)
{
	DVVector inds("uint32_t", n);
	Functor plus("Plus");
	if (!general_scan(n, src_scan, inds, plus, 0)) return (uint32_t)(-1);

	uint32_t ret;
	cuMemcpyDtoH(&ret, (CUdeviceptr)((uint32_t*)inds.data() + n - 1), sizeof(uint32_t));

	static TRTC_For s_for_scatter({ "vec_in1", "vec_in2", "inds", "vec_out1", "vec_out2", "begin_in1", "begin_in2", "begin_out1", "begin_out2" }, "idx",
		"    if ((idx==0 && inds[idx]>0) || (idx>0 && inds[idx]>inds[idx-1]))\n"
		"    {\n"
		"        vec_out1[inds[idx]-1 + begin_out1]=vec_in1[idx +begin_in1];\n"
		"        vec_out2[inds[idx]-1 + begin_out2]=vec_in2[idx +begin_in2];\n"
		"    }\n"
	);

	DVSizeT dvbegin_in1(begin_in1);
	DVSizeT dvbegin_in2(begin_in2);
	DVSizeT dvbegin_out1(begin_out1);
	DVSizeT dvbegin_out2(begin_out2);
	const DeviceViewable* args[] = { &vec_in1, &vec_in2, &inds, &vec_out1, &vec_out2, &dvbegin_in1, &dvbegin_in2, &dvbegin_out1, &dvbegin_out2 };
	if (!s_for_scatter.launch_n(n, args)) return (uint32_t)(-1);
	return ret;
}
