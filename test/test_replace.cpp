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

	int hvec2[5] = { 1, -2, 3, -4, 5 };
	DVVector vec2(ctx, "int32_t", 5, hvec2);
	TRTC_Replace_If(ctx, vec2, { {}, { "x" }, "ret",
		"        ret = x<0;\n" }, DVInt32(0));
	vec2.to_host(hvec2);
	printf("%d %d %d %d %d\n", hvec2[0], hvec2[1], hvec2[2], hvec2[3], hvec2[4]);

	return 0;
}
