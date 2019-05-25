#include <stdio.h>
#include "TRTCContext.h"
#include "fake_vectors/DVCounter.h"
#include "fake_vectors/DVDiscard.h"
#include "transform.h"

int main()
{
	TRTCContext ctx;
	ctx.set_verbose();

	// just to verify that it compiles
	DVDiscard sink(ctx, "int32_t");
	TRTC_Transform(ctx, DVCounter(ctx, DVInt32(5), 10), sink, Functor("Negate"));

	return 0;
}
