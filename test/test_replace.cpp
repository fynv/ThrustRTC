#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "replace.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;

	int hvec[5] = { 1,2,3,1,2 };
	DVVector vec(ctx, "int32_t", 5, hvec);
	TRTC_Replace(ctx, vec, DVInt32(1), DVInt32(99));
	vec.to_host(hvec);
	printf("%d %d %d %d %d\n", hvec[0], hvec[1], hvec[2], hvec[3], hvec[4]);

	return 0;
}
