#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVTransform.h"
#include "transform.h"

int main()
{
	TRTCContext ctx;

	Functor negate = { ctx, {}, { "x" }, "        return -x;\n" };
	Functor square_root{ ctx, {}, { "x" }, "        return sqrtf(x);\n" };

	float hvalues[8] = { 1.0f, 4.0f, 9.0f, 16.0f };
	DVVector dvalues(ctx, "float", 4, hvalues);

	float houtput[4];
	DVVector doutput(ctx, "float", 4);

	DVTransform dtrans(ctx, dvalues, "float", square_root);

	TRTC_Transform(ctx, dtrans, doutput, negate);
	doutput.to_host(houtput);
	printf("%f %f %f %f\n", houtput[0], houtput[1], houtput[2], houtput[3]);

	return 0;
}