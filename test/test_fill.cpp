#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVRange.h"
#include "fill.h"

int main()
{
	DVVector vec_to_fill("int32_t", 10);
	/*
	DVRange front(vec_to_fill, 0, 5);
	DVRange rear(vec_to_fill, 5, 10);*/

	DVVectorAdaptor front(vec_to_fill, 0, 5);
	DVVectorAdaptor rear(vec_to_fill, 5, 10);

	TRTC_Fill(front, DVInt32(123));
	TRTC_Fill(rear, DVInt32(456));

	int values_filled[10];
	vec_to_fill.to_host(values_filled);
	printf("%d %d %d %d %d ", values_filled[0], values_filled[1], values_filled[2], values_filled[3], values_filled[4]);
	printf("%d %d %d %d %d\n", values_filled[5], values_filled[6], values_filled[7], values_filled[8], values_filled[9]);

	return 0;
}
