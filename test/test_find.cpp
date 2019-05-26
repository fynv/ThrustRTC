#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "find.h"

int main()
{
	TRTCContext ctx;
	{
		int h_values[4] = { 0, 5, 3, 7 };
		DVVector d_values(ctx, "int32_t", 4, h_values);
		size_t res;
		TRTC_Find(ctx, d_values, DVInt32(3), res);
		printf("%d\n", res);
		TRTC_Find(ctx, d_values, DVInt32(5), res);
		printf("%d\n", res);
		TRTC_Find(ctx, d_values, DVInt32(9), res);
		printf("%d\n", res);
	}

	{
		int h_values[4] = { 0, 5, 3, 7 };
		DVVector d_values(ctx, "int32_t", 4, h_values);
		size_t res;
		TRTC_Find_If(ctx, d_values, Functor(ctx, {}, { "x" }, "        return x>4;\n"), res);
		printf("%d\n", res);
		TRTC_Find_If(ctx, d_values, Functor(ctx, {}, { "x" }, "        return x>10;\n"), res);
		printf("%d\n", res);	
	}

	{
		int h_values[4] = { 0, 5, 3, 7 };
		DVVector d_values(ctx, "int32_t", 4, h_values);
		size_t res;
		TRTC_Find_If_Not(ctx, d_values, Functor(ctx, {}, { "x" }, "        return x>4;\n"), res);
		printf("%d\n", res);
		TRTC_Find_If_Not(ctx, d_values, Functor(ctx, {}, { "x" }, "        return x>10;\n"), res);
		printf("%d\n", res);
	}
}
