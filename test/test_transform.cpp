#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "transform.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;

	Functor negate = { {},{ "x" }, "ret", "        ret = -x;\n" };

	int hvalues[10] = { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 };
	DVVector vec(ctx, "int32_t", 10, hvalues);
	TRTC_transform(ctx, vec, vec, negate); // in place
	vec.to_host(hvalues);
	printf("%d %d %d %d %d ", hvalues[0], hvalues[1], hvalues[2], hvalues[3], hvalues[4]);
	printf("%d %d %d %d %d\n", hvalues[5], hvalues[6], hvalues[7], hvalues[8], hvalues[9]);

	Functor plus = { {},{ "x", "y" }, "ret", "        ret = x + y;\n" };

	int input1[6] = { -5,  0,  2,  3,  2,  4 };
	int input2[6] = { 3,  6, -2,  1,  2,  3 };
	int output[6];
	DVVector d_in1(ctx, "int32_t", 6, input1);
	DVVector d_in2(ctx, "int32_t", 6, input2);
	DVVector d_out(ctx, "int32_t", 6);
	TRTC_transform(ctx, d_in1, d_in2, d_out, plus); 
	d_out.to_host(output);
	printf("%d %d %d %d %d %d ", output[0], output[1], output[2], output[3], output[4], output[5]);

	return 0;
}
