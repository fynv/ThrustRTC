#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "for_each.h"

int main()
{
	Functor printf_functor = {{}, { "x" }, "        printf(\"%d\\n\", x);\n" };

	int hvec[5] = { 1, 2, 3, 1, 2 };
	DVVector vec("int32_t", 5, hvec);
	TRTC_For_Each(vec, printf_functor);

	return 0;
}
