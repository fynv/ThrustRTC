#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVReverse.h"
#include "transform.h"

int main()
{
	int hinput[4] = { 3, 7, 2, 5 };
	DVVector dinput("int32_t", 4, hinput);

	int houtput[4];
	DVVector doutput("int32_t", 4);

	DVReverse dreverse(dinput);

	TRTC_Transform(dreverse, doutput, Functor("Negate"));
	doutput.to_host(houtput);
	printf("%d %d %d %d\n", houtput[0], houtput[1], houtput[2], houtput[3]);

	return 0;
}
