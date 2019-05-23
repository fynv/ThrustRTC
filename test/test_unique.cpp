#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "unique.h"


int main()
{
	TRTCContext ctx;
	{
		int h_value[7] = { 1, 3, 3, 3, 2, 2, 1 };
		DVVector d_value(ctx, "int32_t", 7, h_value);
		int count = TRTC_Unique(ctx, d_value);
		d_value.to_host(h_value, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_value[i]);
		puts("");
	}

	Functor equal(ctx, {}, { "x", "y" }, "        return x==y;\n");

	{
		int h_value[7] = { 1, 3, 3, 3, 2, 2, 1 };
		DVVector d_value(ctx, "int32_t", 7, h_value);
		int count = TRTC_Unique(ctx, d_value, equal);
		d_value.to_host(h_value, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_value[i]);
		puts("");
	}

	{
		int h_in[7] = { 1, 3, 3, 3, 2, 2, 1 };
		DVVector d_in(ctx, "int32_t", 7, h_in);
		int h_out[7];
		DVVector d_out(ctx, "int32_t", 7);
		int count = TRTC_Unique_Copy(ctx, d_in, d_out);
		d_out.to_host(h_out, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_out[i]);
		puts("");
	}

	{
		int h_in[7] = { 1, 3, 3, 3, 2, 2, 1 };
		DVVector d_in(ctx, "int32_t", 7, h_in);
		int h_out[7];
		DVVector d_out(ctx, "int32_t", 7);
		int count = TRTC_Unique_Copy(ctx, d_in, d_out, equal);
		d_out.to_host(h_out, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_out[i]);
		puts("");
	}

	{
		int h_keys[7] = { 1, 3, 3, 3, 2, 2, 1 };
		int h_values[7] = { 9, 8, 7, 6, 5, 4, 3 };
		DVVector d_keys(ctx, "int32_t", 7, h_keys);
		DVVector d_values(ctx, "int32_t", 7, h_values);
		int count = TRTC_Unique_By_Key(ctx, d_keys, d_values);
		d_keys.to_host(h_keys, 0, count);
		d_values.to_host(h_values, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_keys[i]);
		puts("");
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_values[i]);
		puts("");
	}

	{
		int h_keys[7] = { 1, 3, 3, 3, 2, 2, 1 };
		int h_values[7] = { 9, 8, 7, 6, 5, 4, 3 };
		DVVector d_keys(ctx, "int32_t", 7, h_keys);
		DVVector d_values(ctx, "int32_t", 7, h_values);
		int count = TRTC_Unique_By_Key(ctx, d_keys, d_values, equal);
		d_keys.to_host(h_keys, 0, count);
		d_values.to_host(h_values, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_keys[i]);
		puts("");
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_values[i]);
		puts("");
	}

	{
		int h_keys_in[7] = { 1, 3, 3, 3, 2, 2, 1 };
		int h_values_in[7] = { 9, 8, 7, 6, 5, 4, 3 };
		DVVector d_keys_in(ctx, "int32_t", 7, h_keys_in);
		DVVector d_values_in(ctx, "int32_t", 7, h_values_in);

		int h_keys_out[7];
		int h_values_out[7];
		DVVector d_keys_out(ctx, "int32_t", 7);
		DVVector d_values_out(ctx, "int32_t", 7);

		int count = TRTC_Unique_By_Key_Copy(ctx, d_keys_in, d_values_in, d_keys_out, d_values_out);
		d_keys_out.to_host(h_keys_out, 0, count);
		d_values_out.to_host(h_values_out, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_keys_out[i]);
		puts("");
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_values_out[i]);
		puts("");
	}

	{
		int h_keys_in[7] = { 1, 3, 3, 3, 2, 2, 1 };
		int h_values_in[7] = { 9, 8, 7, 6, 5, 4, 3 };
		DVVector d_keys_in(ctx, "int32_t", 7, h_keys_in);
		DVVector d_values_in(ctx, "int32_t", 7, h_values_in);

		int h_keys_out[7];
		int h_values_out[7];
		DVVector d_keys_out(ctx, "int32_t", 7);
		DVVector d_values_out(ctx, "int32_t", 7);

		int count = TRTC_Unique_By_Key_Copy(ctx, d_keys_in, d_values_in, d_keys_out, d_values_out, equal);
		d_keys_out.to_host(h_keys_out, 0, count);
		d_values_out.to_host(h_values_out, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_keys_out[i]);
		puts("");
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_values_out[i]);
		puts("");
	}

	return 0;
}