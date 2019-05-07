#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "gather.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;

	{
		int values[10] = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
		DVVector d_values(ctx, "int32_t", 10, values);

		int map[10] = { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 };
		DVVector d_map(ctx, "int32_t", 10, map);

		int output[10];
		DVVector d_output(ctx, "int32_t", 10);

		TRTC_Gather(ctx, d_map, d_values, d_output);
		d_output.to_host(output);

		printf("%d %d %d %d %d ", output[0], output[1], output[2], output[3], output[4]);
		printf("%d %d %d %d %d\n", output[5], output[6], output[7], output[8], output[9]);
	}

	{
		int values[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		DVVector d_values(ctx, "int32_t", 10, values);

		int stencil[10] = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
		DVVector d_stencil(ctx, "int32_t", 10, stencil);

		int map[10] = { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 };
		DVVector d_map(ctx, "int32_t", 10, map);

		int output[10] = { 7,7,7,7,7,7,7,7,7,7 };
		DVVector d_output(ctx, "int32_t", 10, output);

		TRTC_Gather_If(ctx, d_map, d_stencil, d_values, d_output);
		d_output.to_host(output);

		printf("%d %d %d %d %d ", output[0], output[1], output[2], output[3], output[4]);
		printf("%d %d %d %d %d\n", output[5], output[6], output[7], output[8], output[9]);

	}

	Functor is_even = { ctx, {},{ "x" }, "        return ((x % 2) == 0);\n" };

	{
		int values[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		DVVector d_values(ctx, "int32_t", 10, values);

		int stencil[10] = { 0, 3, 4, 1, 4, 1, 2, 7, 8, 9 };
		DVVector d_stencil(ctx, "int32_t", 10, stencil);

		int map[10] = { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 };
		DVVector d_map(ctx, "int32_t", 10, map);

		int output[10] = { 7,7,7,7,7,7,7,7,7,7 };
		DVVector d_output(ctx, "int32_t", 10, output);

		TRTC_Gather_If(ctx, d_map, d_stencil, d_values, d_output, is_even);
		d_output.to_host(output);

		printf("%d %d %d %d %d ", output[0], output[1], output[2], output[3], output[4]);
		printf("%d %d %d %d %d\n", output[5], output[6], output[7], output[8], output[9]);
	}

	return 0;
}
