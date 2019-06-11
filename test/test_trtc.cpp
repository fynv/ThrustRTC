#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"

int main()
{
	// kernel with auto parameters, launched twice with different types
	TRTC_Kernel ker(
	{ "arr_in", "arr_out", "k" },
	"    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
	"    if (idx >= arr_in.size()) return;\n"
	"    arr_out[idx] = arr_in[idx]*k;\n");

	float test_f[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0 };
	DVVector dvec_in_f("float", 5, test_f);
	DVVector dvec_out_f("float", 5);
	DVFloat k1(10.0);
	const DeviceViewable* args_f[] = { &dvec_in_f, &dvec_out_f, &k1 };
	ker.launch({ 1, 1, 1 }, { 128, 1, 1 }, args_f);
	dvec_out_f.to_host(test_f);
	printf("%f %f %f %f %f\n", test_f[0], test_f[1], test_f[2], test_f[3], test_f[4]);

	int test_i[5] = { 6, 7, 8, 9, 10 };
	DVVector dvec_in_i("int32_t", 5, test_i);
	DVVector dvec_out_i("int32_t", 5);
	DVInt32 k2(5);
	const DeviceViewable* args_i[] = { &dvec_in_i, &dvec_out_i, &k2 };
	ker.launch({ 1, 1, 1 }, { 128, 1, 1 }, args_i);
	dvec_out_i.to_host(test_i);
	printf("%d %d %d %d %d\n", test_i[0], test_i[1], test_i[2], test_i[3], test_i[4]);

	return 0;
}