#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "transform.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;

	Functor identity = { ctx, {},{ "x" }, "        return x;\n" };
	Functor negate = { ctx, {},{ "x" }, "        return -x;\n" };
	Functor plus = { ctx, {},{ "x", "y" }, "        return x + y;\n" };
	Functor is_odd = { ctx, {},{ "x" }, "        return x % 2;\n" };

	{
		int hvalues[10] = { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 };
		DVVector vec(ctx, "int32_t", 10, hvalues);
		TRTC_Transform(ctx, vec, vec, negate); // in place
		vec.to_host(hvalues);
		printf("%d %d %d %d %d ", hvalues[0], hvalues[1], hvalues[2], hvalues[3], hvalues[4]);
		printf("%d %d %d %d %d\n", hvalues[5], hvalues[6], hvalues[7], hvalues[8], hvalues[9]);
	}

	{
		int input1[6] = { -5,  0,  2,  3,  2,  4 };
		int input2[6] = { 3,  6, -2,  1,  2,  3 };
		int output[6];
		DVVector d_in1(ctx, "int32_t", 6, input1);
		DVVector d_in2(ctx, "int32_t", 6, input2);
		DVVector d_out(ctx, "int32_t", 6);
		TRTC_Transform_Binary(ctx, d_in1, d_in2, d_out, plus);
		d_out.to_host(output);
		printf("%d %d %d %d %d %d\n", output[0], output[1], output[2], output[3], output[4], output[5]);
	}

	{
		int hvalues[10] = { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 };
		DVVector vec(ctx, "int32_t", 10, hvalues);
		TRTC_Transform_If(ctx, vec, vec, negate, is_odd); // in place
		vec.to_host(hvalues);
		printf("%d %d %d %d %d ", hvalues[0], hvalues[1], hvalues[2], hvalues[3], hvalues[4]);
		printf("%d %d %d %d %d\n", hvalues[5], hvalues[6], hvalues[7], hvalues[8], hvalues[9]);
	}

	{
		int data[10] = { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 };
		int stencil[10] = { 1, 0, 1,  0, 1, 0, 1,  0, 1, 0 };
		DVVector d_data(ctx, "int32_t", 10, data);
		DVVector d_stencil(ctx, "int32_t", 10, stencil);
		TRTC_Transform_If_Stencil(ctx, d_data, d_stencil, d_data, negate, identity); // in place
		d_data.to_host(data);
		printf("%d %d %d %d %d ", data[0], data[1], data[2], data[3], data[4]);
		printf("%d %d %d %d %d\n", data[5], data[6], data[7], data[8], data[9]);

	}

	{
		int input1[6] = { -5,  0,  2,  3,  2,  4 };
		int input2[6] = { 3,  6, -2,  1,  2,  3 };
		int stencil[8] = { 1,  0,  1,  0,  1,  0 };
		int output[6] = { -1, -1, -1, -1, -1, -1 };
		DVVector d_in1(ctx, "int32_t", 6, input1);
		DVVector d_in2(ctx, "int32_t", 6, input2);
		DVVector d_stencil(ctx, "int32_t", 6, stencil);
		DVVector d_output(ctx, "int32_t", 6, output);
		TRTC_Transform_Binary_If_Stencil(ctx, d_in1, d_in2, d_stencil, d_output, plus, identity); // in place
		d_output.to_host(output);
		printf("%d %d %d %d %d %d\n", output[0], output[1], output[2], output[3], output[4], output[5]);
	}

	return 0;
}
