#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "sequence.h"
#include "tabulate.h"

int main()
{
	TRTCContext ctx;

	Functor negate = { ctx, {}, { "x" }, "        return -x;\n" };

	int hvalues[10];
	DVVector vec(ctx, "int32_t", 10);
	TRTC_Sequence(ctx, vec);
	TRTC_tabulate(ctx, vec, negate);
	vec.to_host(hvalues);
	printf("%d %d %d %d %d ", hvalues[0], hvalues[1], hvalues[2], hvalues[3], hvalues[4]);
	printf("%d %d %d %d %d\n", hvalues[5], hvalues[6], hvalues[7], hvalues[8], hvalues[9]);

	return 0;
}
