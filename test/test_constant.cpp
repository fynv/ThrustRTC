#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVConstant.h"
#include "transform.h"

int main()
{
	TRTCContext ctx;

	int hvalues[4] = { 3, 7, 2, 5 };
	DVVector vec(ctx, "int32_t", 4, hvalues);

	TRTC_Transform_Binary(ctx, vec, DVConstant(ctx, DVInt32(10)), vec, Functor("Plus"));
	vec.to_host(hvalues);
	printf("%d %d %d %d\n", hvalues[0], hvalues[1], hvalues[2], hvalues[3]);

	return 0;
}
