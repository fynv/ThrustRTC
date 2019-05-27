#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVCounter.h"
#include "scan.h"

int main()
{
	TRTCContext ctx;

	{
		DVCounter d_in(ctx, DVInt32(1), 1000);
		int out[1000];
		DVVector d_out(ctx, "int32_t", 1000);
		TRTC_Inclusive_Scan(ctx, d_in, d_out);
		d_out.to_host(out);
		FILE *fp = fopen("dump.txt", "w");
		for (int i = 0; i < 1000; i++)
			fprintf(fp, "%d\n", out[i]);
		fclose(fp);
	}

	{
		DVCounter d_in(ctx, DVInt32(1), 1000);
		int out[1000];
		DVVector d_out(ctx, "int32_t", 1000);
		TRTC_Exclusive_Scan(ctx, d_in, d_out);
		d_out.to_host(out);
		FILE *fp = fopen("dump2.txt", "w");
		for (int i = 0; i < 1000; i++)
			fprintf(fp, "%d\n", out[i]);
		fclose(fp);
	}

	{
		int data[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector d_data(ctx, "int32_t", 6, data);
		TRTC_Inclusive_Scan(ctx, d_data, d_data);
		d_data.to_host(data);
		printf("%d %d %d %d %d %d\n", data[0], data[1], data[2], data[3], data[4], data[5]);
	}

	{
		int data[10] = { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 };
		DVVector d_data(ctx, "int32_t", 10, data);
		TRTC_Inclusive_Scan(ctx, d_data, d_data, Functor("Maximum"));
		d_data.to_host(data);
		printf("%d %d %d %d %d ", data[0], data[1], data[2], data[3], data[4]);
		printf("%d %d %d %d %d\n", data[5], data[6], data[7], data[8], data[9]);
	}

	{
		int data[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector d_data(ctx, "int32_t", 6, data);
		TRTC_Exclusive_Scan(ctx, d_data, d_data);
		d_data.to_host(data);
		printf("%d %d %d %d %d %d\n", data[0], data[1], data[2], data[3], data[4], data[5]);
	}

	{
		int data[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector d_data(ctx, "int32_t", 6, data);
		TRTC_Exclusive_Scan(ctx, d_data, d_data, DVInt32(4));
		d_data.to_host(data);
		printf("%d %d %d %d %d %d\n", data[0], data[1], data[2], data[3], data[4], data[5]);
	}

	{
		int data[10] = { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 };
		DVVector d_data(ctx, "int32_t", 10, data);
		TRTC_Exclusive_Scan(ctx, d_data, d_data, DVInt32(1), Functor("Maximum"));
		d_data.to_host(data);
		printf("%d %d %d %d %d ", data[0], data[1], data[2], data[3], data[4]);
		printf("%d %d %d %d %d\n", data[5], data[6], data[7], data[8], data[9]);
	}

	return 0;
}
