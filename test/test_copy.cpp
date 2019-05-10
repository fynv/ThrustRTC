#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "copy.h"

int main()
{
	TRTCContext ctx;

	int hIn[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
	DVVector dIn(ctx, "int32_t", 8, hIn);

	int hOut[8];
	DVVector dOut(ctx, "int32_t", 8);
	TRTC_Copy(ctx, dIn, dOut);
	dOut.to_host(hOut);

	printf("%d %d %d %d ", hOut[0], hOut[1], hOut[2], hOut[3]);
	printf("%d %d %d %d\n", hOut[4], hOut[5], hOut[6], hOut[7]);

	return 0;
}
