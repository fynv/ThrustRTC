#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "reduce.h"

int main()
{
	TRTCContext ctx;
	{
		int harr[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector darr(ctx, "int32_t", 6, harr);

		ViewBuf ret;
		TRTC_Reduce(ctx, darr, ret);
		printf("%d\n", *(int*)ret.data());

		TRTC_Reduce(ctx, darr, DVInt32(1), ret);
		printf("%d\n", *(int*)ret.data());

		TRTC_Reduce(ctx, darr, DVInt32(-1), Functor("Maximum"), ret);
		printf("%d\n", *(int*)ret.data());
	}

	{
		int h_keys_in[7] = { 1, 3, 3, 3, 2, 2, 1 };
		DVVector d_keys_in(ctx, "int32_t", 7, h_keys_in);
		int h_values_in[7] = { 9, 8, 7, 6, 5, 4, 3 };
		DVVector d_values_in(ctx, "int32_t", 7, h_values_in);

		int h_keys_out[7];
		DVVector d_keys_out(ctx, "int32_t", 7);
		int h_values_out[7];
		DVVector d_values_out(ctx, "int32_t", 7);

		unsigned count = TRTC_Reduce_By_Key(ctx, d_keys_in, d_values_in, d_keys_out, d_values_out);
		d_keys_out.to_host(h_keys_out, 0, count);
		d_values_out.to_host(h_values_out, 0, count);
		for (unsigned i = 0; i < count; i++)
			printf("%d ", h_keys_out[i]);
		puts("");
		for (unsigned i = 0; i < count; i++)
			printf("%d ", h_values_out[i]);
		puts("");

	}

	{
		int h_keys_in[7] = { 1, 3, 3, 3, 2, 2, 1 };
		DVVector d_keys_in(ctx, "int32_t", 7, h_keys_in);
		int h_values_in[7] = { 9, 8, 7, 6, 5, 4, 3 };
		DVVector d_values_in(ctx, "int32_t", 7, h_values_in);

		int h_keys_out[7];
		DVVector d_keys_out(ctx, "int32_t", 7);
		int h_values_out[7];
		DVVector d_values_out(ctx, "int32_t", 7);

		unsigned count = TRTC_Reduce_By_Key(ctx, d_keys_in, d_values_in, d_keys_out, d_values_out, Functor("EqualTo"));
		d_keys_out.to_host(h_keys_out, 0, count);
		d_values_out.to_host(h_values_out, 0, count);
		for (unsigned i = 0; i < count; i++)
			printf("%d ", h_keys_out[i]);
		puts("");
		for (unsigned i = 0; i < count; i++)
			printf("%d ", h_values_out[i]);
		puts("");

	}

	{
		int h_keys_in[7] = { 1, 3, 3, 3, 2, 2, 1 };
		DVVector d_keys_in(ctx, "int32_t", 7, h_keys_in);
		int h_values_in[7] = { 9, 8, 7, 6, 5, 4, 3 };
		DVVector d_values_in(ctx, "int32_t", 7, h_values_in);

		int h_keys_out[7];
		DVVector d_keys_out(ctx, "int32_t", 7);
		int h_values_out[7];
		DVVector d_values_out(ctx, "int32_t", 7);

		unsigned count = TRTC_Reduce_By_Key(ctx, d_keys_in, d_values_in, d_keys_out, d_values_out, Functor("EqualTo"), Functor("Plus"));
		d_keys_out.to_host(h_keys_out, 0, count);
		d_values_out.to_host(h_values_out, 0, count);
		for (unsigned i = 0; i < count; i++)
			printf("%d ", h_keys_out[i]);
		puts("");
		for (unsigned i = 0; i < count; i++)
			printf("%d ", h_values_out[i]);
		puts("");

	}

	return 0;
}
