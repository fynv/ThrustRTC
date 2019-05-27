#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "binary_search.h"

int main()
{
	TRTCContext ctx;
	{
		int h_values[5] = { 0, 2, 5, 7, 8 };
		DVVector d_values(ctx, "int32_t", 5, h_values);
		size_t res;
		TRTC_Lower_Bound(ctx, d_values, DVInt32(0), res);
		printf("%d\n", (int)res);
		TRTC_Lower_Bound(ctx, d_values, DVInt32(1), res);
		printf("%d\n", (int)res);
		TRTC_Lower_Bound(ctx, d_values, DVInt32(2), res);
		printf("%d\n", (int)res);
		TRTC_Lower_Bound(ctx, d_values, DVInt32(3), res);
		printf("%d\n", (int)res);
		TRTC_Lower_Bound(ctx, d_values, DVInt32(8), res);
		printf("%d\n", (int)res);
		TRTC_Lower_Bound(ctx, d_values, DVInt32(9), res);
		printf("%d\n", (int)res);
	}

}