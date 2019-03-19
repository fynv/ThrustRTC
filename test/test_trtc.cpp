#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");

	TRTCContext ctx;

	TRTCContext::KernelTemplate ktempl(
	{ {"DVVectorView<T>", "arr_in"}, {"DVVectorView<T>", "arr_out"}, {"double", "k"} },
		"    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
		"    if (idx >= arr_in.size) return;"
		"    arr_out.data[idx] = arr_in.data[idx]*k;",
		{ "T" });

	TRTCContext::Kernel* kernel_f = ktempl.instantiate(ctx, { {"T", "float"} });
	float test_f[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0 };
	DVVector dvec_in_f(ctx, "float", test_f, 5);
	DVVector dvec_out_f(ctx, "float", 5);
	DVDouble k1(10.0);
	DeviceViewable* args_f[] = { &dvec_in_f , &dvec_out_f, &k1 };
	TRTCContext::launch_kernel(kernel_f, { 1,1,1 }, { 128,1,1 }, args_f);
	dvec_out_f.ToHost(test_f);
	printf("%f %f %f %f %f\n", test_f[0], test_f[1], test_f[2], test_f[3], test_f[4]);	

	TRTCContext::Kernel* kernel_i = ktempl.instantiate(ctx, { { "T", "int" } });
	int test_i[5] = { 6, 7, 8, 9, 10 };
	DVVector dvec_in_i(ctx, "int", test_i, 5);
	DVVector dvec_out_i(ctx, "int", 5);
	DVDouble k2(5.0);
	DeviceViewable* args_i[] = { &dvec_in_i , &dvec_out_i, &k2 };
	TRTCContext::launch_kernel(kernel_i, { 1,1,1 }, { 128,1,1 }, args_i);
	dvec_out_i.ToHost(test_i);
	printf("%d %d %d %d %d\n", test_i[0], test_i[1], test_i[2], test_i[3], test_i[4]);

	return 0;
}