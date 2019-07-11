#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "sequence.h"
#include "count.h"

int main()
{
	int hin[2000];
	for (int i = 0; i < 2000; i++)
		hin[i] = i % 100;

	DVVector din("int32_t", 2000, hin);
	size_t c;
	TRTC_Count(din, DVInt32(47), c);
	printf("%d\n", (int)c);

	TRTC_Sequence(din);
	Functor op = {{},{ "x" }, "        return (x%100)==47;\n" };
	TRTC_Count_If(din, op, c);
	printf("%d\n", (int)c);

	return 0;
}
