#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVCounter.h"
#include "transform.h"

int main()
{
	int hvalues[10];
	DVVector vec("int32_t", 10);
	TRTC_Transform(DVCounter(DVInt32(5), 10), vec, Functor("Negate"));	
	vec.to_host(hvalues);
	printf("%d %d %d %d %d ", hvalues[0], hvalues[1], hvalues[2], hvalues[3], hvalues[4]);
	printf("%d %d %d %d %d\n", hvalues[5], hvalues[6], hvalues[7], hvalues[8], hvalues[9]);

	return 0;
}
