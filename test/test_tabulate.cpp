#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "sequence.h"
#include "tabulate.h"

int main()
{
	
	int hvalues[10];
	DVVector vec("int32_t", 10);
	TRTC_Sequence(vec);
	TRTC_tabulate(vec, Functor("Negate"));
	vec.to_host(hvalues);
	printf("%d %d %d %d %d ", hvalues[0], hvalues[1], hvalues[2], hvalues[3], hvalues[4]);
	printf("%d %d %d %d %d\n", hvalues[5], hvalues[6], hvalues[7], hvalues[8], hvalues[9]);

	return 0;
}
