#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "merge.h"

int main()
{
	

	{
		int hIn1[6] = { 1, 3, 5, 7, 9, 11 };
		DVVector dIn1("int32_t", 6, hIn1);
		int hIn2[7] = { 1, 1, 2, 3, 5, 8, 13 };
		DVVector dIn2("int32_t", 7, hIn2);
		int hOut[13];
		DVVector dOut("int32_t", 13);
		TRTC_Merge(dIn1, dIn2, dOut);
		dOut.to_host(hOut);
		for (int i = 0; i < 13; i++)
			printf("%d ", hOut[i]);
		puts("");
	}

	{
		int hIn1[6] = { 11, 9, 7, 5, 3, 1 };
		DVVector dIn1("int32_t", 6, hIn1);
		int hIn2[7] = { 13, 8, 5, 3, 2, 1, 1 };
		DVVector dIn2("int32_t", 7, hIn2);
		int hOut[13];
		DVVector dOut("int32_t", 13);
		TRTC_Merge(dIn1, dIn2, dOut, Functor("Greater"));
		dOut.to_host(hOut);
		for (int i = 0; i < 13; i++)
			printf("%d ", hOut[i]);
		puts("");
	}

	{
		int hKeys1[6] = { 1, 3, 5, 7, 9, 11 };
		DVVector dKeys1("int32_t", 6, hKeys1);
		int hVals1[6] = { 0, 0, 0, 0, 0, 0 };
		DVVector dVals1("int32_t", 6, hVals1);

		int hKeys2[7] = { 1, 1, 2, 3, 5, 8, 13 };
		DVVector dKeys2("int32_t", 7, hKeys2);
		int hVals2[7] = { 1, 1, 1, 1, 1, 1, 1 };
		DVVector dVals2("int32_t", 7, hVals2);

		int hKeysOut[13];
		DVVector dKeysOut("int32_t", 13);
		int hValsOut[13];
		DVVector dValsOut("int32_t", 13);

		TRTC_Merge_By_Key(dKeys1, dKeys2, dVals1, dVals2, dKeysOut, dValsOut);
		dKeysOut.to_host(hKeysOut);
		dValsOut.to_host(hValsOut);
		for (int i = 0; i < 13; i++)
			printf("%d ", hKeysOut[i]);
		puts("");
		for (int i = 0; i < 13; i++)
			printf("%d ", hValsOut[i]);
		puts("");
	}

	{
		int hKeys1[6] = { 11, 9, 7, 5, 3, 1 };
		DVVector dKeys1("int32_t", 6, hKeys1);
		int hVals1[6] = { 0, 0, 0, 0, 0, 0 };
		DVVector dVals1("int32_t", 6, hVals1);

		int hKeys2[7] = { 13, 8, 5, 3, 2, 1, 1 };
		DVVector dKeys2("int32_t", 7, hKeys2);
		int hVals2[7] = { 1, 1, 1, 1, 1, 1, 1 };
		DVVector dVals2("int32_t", 7, hVals2);

		int hKeysOut[13];
		DVVector dKeysOut("int32_t", 13);
		int hValsOut[13];
		DVVector dValsOut("int32_t", 13);

		TRTC_Merge_By_Key(dKeys1, dKeys2, dVals1, dVals2, dKeysOut, dValsOut, Functor("Greater"));
		dKeysOut.to_host(hKeysOut);
		dValsOut.to_host(hValsOut);
		for (int i = 0; i < 13; i++)
			printf("%d ", hKeysOut[i]);
		puts("");
		for (int i = 0; i < 13; i++)
			printf("%d ", hValsOut[i]);
		puts("");
	}

	return 0;
}
