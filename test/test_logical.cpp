#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "logical.h"

int main()
{
	TRTCContext ctx;

	Functor identity("Identity");

	bool A[3] = { true, true, false };
	DVVector d_A(ctx, "bool", 3, A);

	bool res;

	TRTC_All_Of(ctx, d_A, identity, res, 0, 2);
	printf("%d\n", res);
	TRTC_All_Of(ctx, d_A, identity, res, 0, 3);
	printf("%d\n", res);
	TRTC_All_Of(ctx, d_A, identity, res, 0, 0);
	printf("%d\n", res);

	TRTC_Any_Of(ctx, d_A, identity, res, 0, 2);
	printf("%d\n", res);
	TRTC_Any_Of(ctx, d_A, identity, res, 0, 3);
	printf("%d\n", res);
	TRTC_Any_Of(ctx, d_A, identity, res, 2, 3);
	printf("%d\n", res);
	TRTC_Any_Of(ctx, d_A, identity, res, 0, 0);
	printf("%d\n", res);

	TRTC_None_Of(ctx, d_A, identity, res, 0, 2);
	printf("%d\n", res);
	TRTC_None_Of(ctx, d_A, identity, res, 0, 3);
	printf("%d\n", res);
	TRTC_None_Of(ctx, d_A, identity, res, 2, 3);
	printf("%d\n", res);
	TRTC_None_Of(ctx, d_A, identity, res, 0, 0);
	printf("%d\n", res);

	return 0;
}