#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVConstant.h"
#include "transform.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;

	Functor plus = { {},{ "x", "y" }, "ret", "        ret = x + y;\n" };

	int hvalues[4] = { 3, 7, 2, 5 };
	DVVector vec(ctx, "int32_t", 4, hvalues);

	TRTC_Transform_Binary(ctx, vec, DVConstant(ctx, DVInt32(10)), vec, plus);
	vec.to_host(hvalues);
	printf("%d %d %d %d\n", hvalues[0], hvalues[1], hvalues[2], hvalues[3]);

	return 0;
}
