#include <stdio.h>
#include "TRTCContext.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");

	TRTCContext ctx;
	printf("%d %d\n", ctx.size_of("float"), ctx.size_of("double"));

	return 0;
}