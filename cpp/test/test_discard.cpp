#include <stdio.h>
#include "TRTCContext.h"
#include "fake_vectors/DVCounter.h"
#include "fake_vectors/DVDiscard.h"
#include "transform.h"

int main()
{	
	TRTC_Set_Verbose();

	// just to verify that it compiles
	DVDiscard sink("int32_t");
	TRTC_Transform(DVCounter(DVInt32(5), 10), sink, Functor("Negate"));

	return 0;
}
