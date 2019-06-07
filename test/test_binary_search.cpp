#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "binary_search.h"

int main()
{
	TRTCContext ctx;

	int h_input[5] = { 0, 2, 5, 7, 8 };
	DVVector d_input(ctx, "int32_t", 5, h_input);
	{
		size_t res;
		TRTC_Lower_Bound(ctx, d_input, DVInt32(0), res);
		printf("%d\n", (int)res);
		TRTC_Lower_Bound(ctx, d_input, DVInt32(1), res);
		printf("%d\n", (int)res);
		TRTC_Lower_Bound(ctx, d_input, DVInt32(2), res);
		printf("%d\n", (int)res);
		TRTC_Lower_Bound(ctx, d_input, DVInt32(3), res);
		printf("%d\n", (int)res);
		TRTC_Lower_Bound(ctx, d_input, DVInt32(8), res);
		printf("%d\n", (int)res);
		TRTC_Lower_Bound(ctx, d_input, DVInt32(9), res);
		printf("%d\n", (int)res);
	}
	puts("");

	{
		size_t res;
		TRTC_Upper_Bound(ctx, d_input, DVInt32(0), res);
		printf("%d\n", (int)res);
		TRTC_Upper_Bound(ctx, d_input, DVInt32(1), res);
		printf("%d\n", (int)res);
		TRTC_Upper_Bound(ctx, d_input, DVInt32(2), res);
		printf("%d\n", (int)res);
		TRTC_Upper_Bound(ctx, d_input, DVInt32(3), res);
		printf("%d\n", (int)res);
		TRTC_Upper_Bound(ctx, d_input, DVInt32(8), res);
		printf("%d\n", (int)res);
		TRTC_Upper_Bound(ctx, d_input, DVInt32(9), res);
		printf("%d\n", (int)res);
	}
	puts("");

	{
		bool res;
		TRTC_Binary_Search(ctx, d_input, DVInt32(0), res);
		puts(res ? "true" : "false");
		TRTC_Binary_Search(ctx, d_input, DVInt32(1), res);
		puts(res ? "true" : "false");
		TRTC_Binary_Search(ctx, d_input, DVInt32(2), res);
		puts(res ? "true" : "false");
		TRTC_Binary_Search(ctx, d_input, DVInt32(3), res);
		puts(res ? "true" : "false");
		TRTC_Binary_Search(ctx, d_input, DVInt32(8), res);
		puts(res ? "true" : "false");
		TRTC_Binary_Search(ctx, d_input, DVInt32(9), res);
		puts(res ? "true" : "false");
	}
	puts("");

	{
		int h_values[6] = { 0, 1, 2, 3, 8, 9 };
		DVVector d_values(ctx, "int32_t", 6, h_values);

		int h_output[6];
		DVVector d_output(ctx, "int32_t", 6);

		TRTC_Lower_Bound_V(ctx, d_input, d_values, d_output);

		d_output.to_host(h_output);
		printf("%d %d %d %d %d %d ", h_output[0], h_output[1], h_output[2], h_output[3], h_output[4], h_output[5]);
	}
	puts("");

	{
		int h_values[6] = { 0, 1, 2, 3, 8, 9 };
		DVVector d_values(ctx, "int32_t", 6, h_values);

		int h_output[6];
		DVVector d_output(ctx, "int32_t", 6);

		TRTC_Upper_Bound_V(ctx, d_input, d_values, d_output);

		d_output.to_host(h_output);
		printf("%d %d %d %d %d %d ", h_output[0], h_output[1], h_output[2], h_output[3], h_output[4], h_output[5]);
	}
	puts("");

	{
		int h_values[6] = { 0, 1, 2, 3, 8, 9 };
		DVVector d_values(ctx, "int32_t", 6, h_values);

		int h_output[6];
		DVVector d_output(ctx, "int32_t", 6);

		TRTC_Binary_Search_V(ctx, d_input, d_values, d_output);

		d_output.to_host(h_output);
		printf("%d %d %d %d %d %d ", h_output[0], h_output[1], h_output[2], h_output[3], h_output[4], h_output[5]);
	}
	puts("");

}