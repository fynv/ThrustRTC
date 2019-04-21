#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "swap.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;

	int harr1[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
	DVVector darr1(ctx, "int32_t", 8, harr1);

	int harr2[8] = { 1000, 900, 800, 700, 600, 500, 400, 300 };
	DVVector darr2(ctx, "int32_t", 8, harr2);

	TRTC_Swap(ctx, darr1, darr2);
	darr1.to_host(harr1);
	darr2.to_host(harr2);

	printf("%d %d %d %d ", harr1[0], harr1[1], harr1[2], harr1[3]);
	printf("%d %d %d %d\n", harr1[4], harr1[5], harr1[6], harr1[7]);
	printf("%d %d %d %d ", harr2[0], harr2[1], harr2[2], harr2[3]);
	printf("%d %d %d %d\n", harr2[4], harr2[5], harr2[6], harr2[7]);

	return 0;
}
