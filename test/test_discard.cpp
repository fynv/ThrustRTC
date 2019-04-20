#include <stdio.h>
#include "TRTCContext.h"
#include "fake_vectors/DVCounter.h"
#include "fake_vectors/DVDiscard.h"
#include "transform.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;
	ctx.set_verbose();

	Functor negate = { {}, { "x" }, "ret", "        ret = -x;\n" };
	
	// just to verify that it compiles
	DVDiscard sink(ctx, "int32_t");
	TRTC_Transform(ctx, DVCounter(ctx, DVInt32(5), 10), sink, negate);	

	return 0;
}
