#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "transform_reduce.h"

int main()
{

	Functor absolute_value = { {},{ "x" }, "        return x<(decltype(x))0 ? -x : x;\n" };

	int h_data[6] = { -1, 0, -2, -2, 1, -3 };
	DVVector d_data("int32_t", 6, h_data);

	ViewBuf res;
	TRTC_Transform_Reduce(d_data, absolute_value, DVInt32(0), Functor("Maximum"), res);
	printf("%d\n", *(int*)res.data());


	return 0;
}
