#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "mismatch.h"

int main()
{	
	{
		int h1[4] = { 0, 5, 3, 7 };
		DVVector d1("int32_t", 4, h1);
		int h2[4] = { 0, 5, 8, 7 };
		DVVector d2("int32_t", 4, h2);
		size_t res;
		TRTC_Mismatch(d1, d2, res);
		printf("%d\n", (int)res);
	}

	{
		int h1[4] = { 0, 5, 3, 7 };
		DVVector d1("int32_t", 4, h1);
		int h2[4] = { 0, 5, 8, 7 };
		DVVector d2("int32_t", 4, h2);
		size_t res;
		TRTC_Mismatch(d1, d2, Functor("EqualTo"), res);
		printf("%d\n", (int)res);
	}

}