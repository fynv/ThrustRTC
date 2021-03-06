#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVPermutation.h"
#include "transform.h"

int main()
{
	float hvalues[8] = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f };
	DVVector dvalues("float", 8, hvalues);

	int hindices[4] = { 2,6,1,3 };
	DVVector dindices("int32_t", 4, hindices);

	float houtput[4];
	DVVector doutput("float", 4);

	DVPermutation perm(dvalues, dindices);

	TRTC_Transform(perm, doutput, Functor("Negate"));
	doutput.to_host(houtput);
	printf("%f %f %f %f\n", houtput[0], houtput[1], houtput[2], houtput[3]);

	return 0;
}
