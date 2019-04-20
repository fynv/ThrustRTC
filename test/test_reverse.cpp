#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVReverse.h"
#include "transform.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;

	Functor negate = { {}, { "x" }, "ret", "        ret = -x;\n" };

	int hinput[4] = { 3, 7, 2, 5 };
	DVVector dinput(ctx, "int32_t", 4, hinput);

	int houtput[4];
	DVVector doutput(ctx, "int32_t", 4);

	DVReverse dreverse(ctx, dinput);

	TRTC_Transform(ctx, dreverse, doutput, negate);
	doutput.to_host(houtput);
	printf("%d %d %d %d\n", houtput[0], houtput[1], houtput[2], houtput[3]);

	return 0;
}
