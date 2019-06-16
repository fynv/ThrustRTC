#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "copy.h"

int main()
{

	{
		int hIn[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
		DVVector dIn("int32_t", 8, hIn);

		int hOut[8];
		DVVector dOut("int32_t", 8);
		TRTC_Copy(dIn, dOut);
		dOut.to_host(hOut);

		printf("%d %d %d %d ", hOut[0], hOut[1], hOut[2], hOut[3]);
		printf("%d %d %d %d\n", hOut[4], hOut[5], hOut[6], hOut[7]);
	}

	Functor is_even = { {},{ "x" }, "        return x % 2 == 0;\n" };

	{
		int hIn[6] = { -2, 0, -1, 0, 1, 2 };
		DVVector dIn("int32_t", 6, hIn);
		int hOut[6];
		DVVector dOut("int32_t", 6);
		uint32_t count = TRTC_Copy_If(dIn, dOut, is_even);
		dOut.to_host(hOut, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", hOut[i]);
		puts("");
	}

	{
		int hIn[6] = { 0, 1, 2, 3, 4, 5 };
		DVVector dIn("int32_t", 6, hIn);
		int hStencil[6] = { -2, 0, -1, 0, 1, 2 };
		DVVector dStencil("int32_t", 6, hStencil);

		int hOut[6];
		DVVector dOut("int32_t", 6);
		uint32_t count = TRTC_Copy_If_Stencil(dIn, dStencil, dOut, is_even);
		dOut.to_host(hOut, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", hOut[i]);
		puts("");
	}


	return 0;
}
