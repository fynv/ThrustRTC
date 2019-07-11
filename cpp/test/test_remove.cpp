#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "remove.h"

int main()
{
	
	{
		int h_value[6] = { 3, 1, 4, 1, 5, 9 };
		DVVector d_value("int32_t", 6, h_value);
		uint32_t count = TRTC_Remove(d_value, DVInt32(1));
		d_value.to_host(h_value, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_value[i]);
		puts("");
	}

	{
		int h_in[6] = { -2, 0, -1, 0, 1, 2 };
		DVVector d_in("int32_t", 6, h_in);
		int h_out[6];
		DVVector d_out("int32_t", 6);
		uint32_t count = TRTC_Remove_Copy(d_in, d_out, DVInt32(0));
		d_out.to_host(h_out, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_out[i]);
		puts("");
	}

	Functor is_even = { {},{ "x" }, "        return x % 2 == 0;\n" };

	{
		int h_value[6] = { 1, 4, 2, 8, 5, 7 };
		DVVector d_value("int32_t", 6, h_value);
		uint32_t count = TRTC_Remove_If(d_value, is_even);
		d_value.to_host(h_value, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_value[i]);
		puts("");
	}

	{
		int h_in[6] = { -2, 0, -1, 0, 1, 2 };
		DVVector d_in("int32_t", 6, h_in);
		int h_out[6];
		DVVector d_out("int32_t", 6);
		uint32_t count = TRTC_Remove_Copy_If(d_in, d_out, is_even);
		d_out.to_host(h_out, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_out[i]);
		puts("");
	}

	Functor identity("Identity");

	{
		int h_value[6] = { 1, 4, 2, 8, 5, 7 };
		DVVector d_value("int32_t", 6, h_value);
		int h_stencil[6] = { 0, 1, 1, 1, 0, 0 };
		DVVector d_stencil("int32_t", 6, h_stencil);
		uint32_t count = TRTC_Remove_If_Stencil(d_value, d_stencil, identity);
		d_value.to_host(h_value, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_value[i]);
		puts("");
	}

	{
		int h_in[6] = { -2, 0, -1, 0, 1, 2 };
		DVVector d_in("int32_t", 6, h_in);
		int h_stencil[6] = { 1, 1, 0, 1, 0, 1 };
		DVVector d_stencil("int32_t", 6, h_stencil);
		int h_out[6];
		DVVector d_out("int32_t", 6);
		uint32_t count = TRTC_Remove_Copy_If_Stencil(d_in, d_stencil, d_out, identity);
		d_out.to_host(h_out, 0, count);
		for (uint32_t i = 0; i < count; i++)
			printf("%d ", h_out[i]);
		puts("");
	}

	return 0;
}