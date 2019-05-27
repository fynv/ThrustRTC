#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "transform_reduce.h"

int main()
{
	TRTCContext ctx;

	Functor absolute_value = { ctx, {},{ "x" }, "        return x<(decltype(x))0 ? -x : x;\n" };

	int h_data[6] = { -1, 0, -2, -2, 1, -3 };
	DVVector d_data(ctx, "int32_t", 6, h_data);

	ViewBuf res;
	TRTC_Transform_Reduce(ctx, d_data, absolute_value, DVInt32(0), Functor("Maximum"), res);
	printf("%d\n", *(int*)res.data());


	return 0;
}
