#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "sequence.h"
#include "reduce.h"
#include "fake_vectors/DVCustomVector.h"
#include "fake_vectors/DVDiscard.h"

int main()
{
	

	DVVector din("int32_t", 2000);
	TRTC_Sequence(din);

	DVCustomVector d_custom_values({ {"src", &din} }, "idx",
		"    unsigned group = idx / src.size();\n"
		"    unsigned sub_idx = idx % src.size();\n"
		"    return src[sub_idx] % (group+1) ==0 ? 1: 0;\n ", "uint32_t", din.size() * 10);

	DVCustomVector d_custom_keys({ {"src", &din} }, "idx", 
		"    return idx / src.size();\n", "uint32_t", din.size() * 10);

	DVVector d_values_out("uint32_t", 10);
	DVDiscard d_keys_out("uint32_t", 10);

	TRTC_Reduce_By_Key(d_custom_keys, d_custom_values, d_keys_out, d_values_out);

	unsigned out[10];
	d_values_out.to_host(out);
	for (int i = 0; i < 10; i++)
		printf("%d ", out[i]);
	puts("");

	return 0;
}

