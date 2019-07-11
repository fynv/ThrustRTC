#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "DVTuple.h"
#include "functor.h"
#include "fake_vectors//DVConstant.h"
#include "fake_vectors/DVCounter.h"
#include "fake_vectors/DVTransform.h"
#include "fake_vectors/DVZipped.h"
#include "copy.h"

int main()
{
	
	{
		int h_int_in[5] = { 0, 1, 2, 3, 4};
		DVVector d_int_in("int32_t", 5, h_int_in);
		float h_float_in[5] = { 0.0f, 10.0f, 20.0f, 30.0f, 40.0f };
		DVVector d_float_in("float", 5, h_float_in);

		int h_int_out[5];
		DVVector d_int_out("int32_t", 5);
		float h_float_out[5];
		DVVector d_float_out("float", 5);

		DVZipped zipped_in({ &d_int_in, &d_float_in }, { "a", "b" });
		DVZipped zipped_out({ &d_int_out, &d_float_out }, { "a", "b" });

		TRTC_Copy(zipped_in, zipped_out);
		d_int_out.to_host(h_int_out);
		d_float_out.to_host(h_float_out);

		printf("%d %d %d %d %d\n", h_int_out[0], h_int_out[1], h_int_out[2], h_int_out[3], h_int_out[4]);
		printf("%f %f %f %f %f\n", h_float_out[0], h_float_out[1], h_float_out[2], h_float_out[3], h_float_out[4]);

	}

	{
		DVCounter d_int_in(DVInt32(0), 5);
		DVTransform d_float_in(d_int_in, "float", Functor({}, { "i" }, "        return (float)i*10.0f +10.0f;\n"));

		int h_int_out[5];
		DVVector d_int_out("int32_t", 5);
		float h_float_out[5];
		DVVector d_float_out("float", 5);

		DVZipped zipped_in({ &d_int_in, &d_float_in }, { "a", "b" });
		DVZipped zipped_out({ &d_int_out, &d_float_out }, { "a", "b" });

		TRTC_Copy(zipped_in, zipped_out);
		d_int_out.to_host(h_int_out);
		d_float_out.to_host(h_float_out);

		printf("%d %d %d %d %d\n", h_int_out[0], h_int_out[1], h_int_out[2], h_int_out[3], h_int_out[4]);
		printf("%f %f %f %f %f\n", h_float_out[0], h_float_out[1], h_float_out[2], h_float_out[3], h_float_out[4]);

	}

	{
		DVInt32 d_int_in(123);
		DVFloat d_float_in(456.0f);
		DVTuple d_tuple({ {"a", &d_int_in}, {"b",&d_float_in} });
		DVConstant const_in(d_tuple, 5);

		int h_int_out[5];
		DVVector d_int_out("int32_t", 5);
		float h_float_out[5];
		DVVector d_float_out("float", 5);
		DVZipped zipped_out({ &d_int_out, &d_float_out }, { "a", "b" });

		TRTC_Copy(const_in, zipped_out);
		d_int_out.to_host(h_int_out);
		d_float_out.to_host(h_float_out);

		printf("%d %d %d %d %d\n", h_int_out[0], h_int_out[1], h_int_out[2], h_int_out[3], h_int_out[4]);
		printf("%f %f %f %f %f\n", h_float_out[0], h_float_out[1], h_float_out[2], h_float_out[3], h_float_out[4]);

	}

	return 0;
}

