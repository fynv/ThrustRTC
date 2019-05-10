#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "scatter.h"

int main()
{
	TRTCContext ctx;

	{
		int values[10] = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
		DVVector d_values(ctx, "int32_t", 10, values);

		int map[10] = { 0, 5, 1, 6, 2, 7, 3, 8, 4, 9 };
		DVVector d_map(ctx, "int32_t", 10, map);

		int output[10];
		DVVector d_output(ctx, "int32_t", 10);

		TRTC_Scatter(ctx, d_values, d_map, d_output);
		d_output.to_host(output);

		printf("%d %d %d %d %d ", output[0], output[1], output[2], output[3], output[4]);
		printf("%d %d %d %d %d\n", output[5], output[6], output[7], output[8], output[9]);
	}

	{
		int h_V[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
		DVVector d_V(ctx, "int32_t", 8, h_V);

		int h_M[8] = { 0, 5, 1, 6, 2, 7, 3, 4 };
		DVVector d_M(ctx, "int32_t", 8, h_M);

		int h_S[8] = { 1, 0, 1, 0, 1, 0, 1, 0 };
		DVVector d_S(ctx, "int32_t", 8, h_S);

		int h_D[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		DVVector d_D(ctx, "int32_t", 8, h_D);

		TRTC_Scatter_If(ctx, d_V, d_M, d_S, d_D);
		d_D.to_host(h_D);

		printf("%d %d %d %d ", h_D[0], h_D[1], h_D[2], h_D[3]);
		printf("%d %d %d %d\n", h_D[4], h_D[5], h_D[6], h_D[7]);

	}

	Functor is_even = { ctx, {},{ "x" }, "        return ((x % 2) == 0);\n" };

	{
		int h_V[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
		DVVector d_V(ctx, "int32_t", 8, h_V);

		int h_M[8] = { 0, 5, 1, 6, 2, 7, 3, 4 };
		DVVector d_M(ctx, "int32_t", 8, h_M);

		int h_S[8] = { 2, 1, 2, 1, 2, 1, 2, 1 };
		DVVector d_S(ctx, "int32_t", 8, h_S);

		int h_D[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		DVVector d_D(ctx, "int32_t", 8, h_D);

		TRTC_Scatter_If(ctx, d_V, d_M, d_S, d_D, is_even);
		d_D.to_host(h_D);

		printf("%d %d %d %d ", h_D[0], h_D[1], h_D[2], h_D[3]);
		printf("%d %d %d %d\n", h_D[4], h_D[5], h_D[6], h_D[7]);
	}

	return 0;
}
