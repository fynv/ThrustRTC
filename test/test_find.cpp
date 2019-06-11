#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "find.h"

int main()
{
	
	{
		int h_values[4] = { 0, 5, 3, 7 };
		DVVector d_values("int32_t", 4, h_values);
		size_t res;
		TRTC_Find(d_values, DVInt32(3), res);
		printf("%d\n", (int)res);
		TRTC_Find(d_values, DVInt32(5), res);
		printf("%d\n", (int)res);
		TRTC_Find(d_values, DVInt32(9), res);
		printf("%d\n", (int)res);
	}

	{
		int h_values[4] = { 0, 5, 3, 7 };
		DVVector d_values("int32_t", 4, h_values);
		size_t res;
		TRTC_Find_If(d_values, Functor({}, { "x" }, "        return x>4;\n"), res);
		printf("%d\n", (int)res);
		TRTC_Find_If(d_values, Functor({}, { "x" }, "        return x>10;\n"), res);
		printf("%d\n", (int)res);	
	}

	{
		int h_values[4] = { 0, 5, 3, 7 };
		DVVector d_values("int32_t", 4, h_values);
		size_t res;
		TRTC_Find_If_Not(d_values, Functor({}, { "x" }, "        return x>4;\n"), res);
		printf("%d\n", (int)res);
		TRTC_Find_If_Not(d_values, Functor({}, { "x" }, "        return x>10;\n"), res);
		printf("%d\n", (int)res);
	}
}
