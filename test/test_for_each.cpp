#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "for_each.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;

	Functor printf_functor = { {}, { "x" }, nullptr, "        printf(\"%d\\n\", x);\n" };

	int hvec[5] = { 1, 2, 3, 1, 2 };
	DVVector vec(ctx, "int32_t", 5, hvec);
	TRTC_For_Each(ctx, vec, printf_functor);

	return 0;
}
