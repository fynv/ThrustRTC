#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "adjacent_difference.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;

	Functor plus = { {}, { "lhs", "rhs" }, "ret", "            ret = lhs + rhs;\n" };

	int hvec1[8] = { 1, 2, 1, 2, 1, 2, 1, 2 };
	DVVector vec1_in(ctx, "int32_t", 8, hvec1);
	DVVector vec1_out(ctx, "int32_t", 8);
	TRTC_Adjacent_Difference(ctx, vec1_in, vec1_out);
	vec1_out.to_host(hvec1);
	printf("%d %d %d %d %d %d %d %d\n", hvec1[0], hvec1[1], hvec1[2], hvec1[3], hvec1[4], hvec1[5], hvec1[6], hvec1[7]);

	int hvec2[8] = { 1, 2, 1, 2, 1, 2, 1, 2 };
	DVVector vec2_in(ctx, "int32_t", 8, hvec2);
	DVVector vec2_out(ctx, "int32_t", 8);
	TRTC_Adjacent_Difference(ctx, vec2_in, vec2_out, plus);
	vec2_out.to_host(hvec2);
	printf("%d %d %d %d %d %d %d %d\n", hvec2[0], hvec2[1], hvec2[2], hvec2[3], hvec2[4], hvec2[5], hvec2[6], hvec2[7]);
}