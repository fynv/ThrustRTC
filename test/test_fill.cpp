#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "fill.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;

	DVVector vec_to_fill(ctx, "int32_t", 5);
	TRTC_Fill(ctx, vec_to_fill, DVInt32(123));
	int values_filled[5];
	vec_to_fill.to_host(values_filled);
	printf("%d %d %d %d %d\n", values_filled[0], values_filled[1], values_filled[2], values_filled[3], values_filled[4]);

	return 0;
}