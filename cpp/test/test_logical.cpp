#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "logical.h"

int main()
{
	Functor identity("Identity");

	bool A[3] = { true, true, false };
	DVVector d_A("bool", 3, A);

	bool res;

	TRTC_All_Of(DVVectorAdaptor(d_A, 0, 2), identity, res);
	printf("%d\n", res);
	TRTC_All_Of(DVVectorAdaptor(d_A, 0, 3), identity, res);
	printf("%d\n", res);
	TRTC_All_Of(DVVectorAdaptor(d_A, 0, 0), identity, res);
	printf("%d\n", res);

	TRTC_Any_Of(DVVectorAdaptor(d_A, 0, 2), identity, res);
	printf("%d\n", res);
	TRTC_Any_Of(DVVectorAdaptor(d_A, 0, 3), identity, res);
	printf("%d\n", res);
	TRTC_Any_Of(DVVectorAdaptor(d_A, 2, 3), identity, res);
	printf("%d\n", res);
	TRTC_Any_Of(DVVectorAdaptor(d_A, 0, 0), identity, res);
	printf("%d\n", res);

	TRTC_None_Of(DVVectorAdaptor(d_A, 0, 2), identity, res);
	printf("%d\n", res);
	TRTC_None_Of(DVVectorAdaptor(d_A, 0, 3), identity, res);
	printf("%d\n", res);
	TRTC_None_Of(DVVectorAdaptor(d_A, 2, 3), identity, res);
	printf("%d\n", res);
	TRTC_None_Of(DVVectorAdaptor(d_A, 0, 0), identity, res);
	printf("%d\n", res);

	return 0;
}