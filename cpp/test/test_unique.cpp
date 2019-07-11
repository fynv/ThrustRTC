#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "unique.h"


int main()
{	
	{
		int h_value[7] = { 1, 3, 3, 3, 2, 2, 1 };
		DVVector d_value("int32_t", 7, h_value);
		int count = TRTC_Unique(d_value);
		d_value.to_host(h_value, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_value[i]);
		puts("");
	}

	{
		int h_value[7] = { 1, 3, 3, 3, 2, 2, 1 };
		DVVector d_value("int32_t", 7, h_value);
		int count = TRTC_Unique(d_value, Functor("EqualTo"));
		d_value.to_host(h_value, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_value[i]);
		puts("");
	}

	{
		int h_in[7] = { 1, 3, 3, 3, 2, 2, 1 };
		DVVector d_in("int32_t", 7, h_in);
		int h_out[7];
		DVVector d_out("int32_t", 7);
		int count = TRTC_Unique_Copy(d_in, d_out);
		d_out.to_host(h_out, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_out[i]);
		puts("");
	}

	{
		int h_in[7] = { 1, 3, 3, 3, 2, 2, 1 };
		DVVector d_in("int32_t", 7, h_in);
		int h_out[7];
		DVVector d_out("int32_t", 7);
		int count = TRTC_Unique_Copy(d_in, d_out, Functor("EqualTo"));
		d_out.to_host(h_out, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_out[i]);
		puts("");
	}

	{
		int h_keys[7] = { 1, 3, 3, 3, 2, 2, 1 };
		int h_values[7] = { 9, 8, 7, 6, 5, 4, 3 };
		DVVector d_keys("int32_t", 7, h_keys);
		DVVector d_values("int32_t", 7, h_values);
		int count = TRTC_Unique_By_Key(d_keys, d_values);
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
		DVVector d_keys("int32_t", 7, h_keys);
		DVVector d_values("int32_t", 7, h_values);
		int count = TRTC_Unique_By_Key(d_keys, d_values, Functor("EqualTo"));
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
		DVVector d_keys_in("int32_t", 7, h_keys_in);
		DVVector d_values_in("int32_t", 7, h_values_in);

		int h_keys_out[7];
		int h_values_out[7];
		DVVector d_keys_out("int32_t", 7);
		DVVector d_values_out("int32_t", 7);

		int count = TRTC_Unique_By_Key_Copy(d_keys_in, d_values_in, d_keys_out, d_values_out);
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
		DVVector d_keys_in("int32_t", 7, h_keys_in);
		DVVector d_values_in("int32_t", 7, h_values_in);

		int h_keys_out[7];
		int h_values_out[7];
		DVVector d_keys_out("int32_t", 7);
		DVVector d_values_out("int32_t", 7);

		int count = TRTC_Unique_By_Key_Copy(d_keys_in, d_values_in, d_keys_out, d_values_out, Functor("EqualTo"));
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
