#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "scan.h"

int main()
{
	
	{
		int h_keys[1000];
		int h_values[1000];
		for (int i = 0; i < 1000; i++)
		{
			h_keys[i] = i / 300;
			h_values[i] = i % 300;
		}
		DVVector d_keys("int32_t", 1000, h_keys);
		DVVector d_values("int32_t", 1000, h_values);
		TRTC_Inclusive_Scan_By_Key(d_keys, d_values, d_values);
		d_values.to_host(h_values);
		FILE *fp = fopen("dump.txt", "w");
		for (int i = 0; i < 1000; i++)
			fprintf(fp, "%d, %d\n", h_keys[i], h_values[i]);
		fclose(fp);
	}

	{
		int h_keys[10]= { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3 };
		int h_values[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
		DVVector d_keys("int32_t", 10, h_keys);
		DVVector d_values("int32_t", 10, h_values);
		TRTC_Inclusive_Scan_By_Key(d_keys, d_values, d_values);
		d_values.to_host(h_values);
		printf("%d %d %d %d %d ", h_values[0], h_values[1], h_values[2], h_values[3], h_values[4]);
		printf("%d %d %d %d %d\n", h_values[5], h_values[6], h_values[7], h_values[8], h_values[9]);
	}

	{
		int h_keys[10] = { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3 };
		int h_values[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
		DVVector d_keys("int32_t", 10, h_keys);
		DVVector d_values("int32_t", 10, h_values);
		TRTC_Exclusive_Scan_By_Key(d_keys, d_values, d_values);
		d_values.to_host(h_values);
		printf("%d %d %d %d %d ", h_values[0], h_values[1], h_values[2], h_values[3], h_values[4]);
		printf("%d %d %d %d %d\n", h_values[5], h_values[6], h_values[7], h_values[8], h_values[9]);
	}

	{
		int h_keys[10] = { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3 };
		int h_values[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
		DVVector d_keys("int32_t", 10, h_keys);
		DVVector d_values("int32_t", 10, h_values);
		TRTC_Exclusive_Scan_By_Key(d_keys, d_values, d_values, DVInt32(5));
		d_values.to_host(h_values);
		printf("%d %d %d %d %d ", h_values[0], h_values[1], h_values[2], h_values[3], h_values[4]);
		printf("%d %d %d %d %d\n", h_values[5], h_values[6], h_values[7], h_values[8], h_values[9]);
	}

	{
		int h_keys[10] = { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3 };
		int h_values[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
		DVVector d_keys("int32_t", 10, h_keys);
		DVVector d_values("int32_t", 10, h_values);
		TRTC_Exclusive_Scan_By_Key(d_keys, d_values, d_values, DVInt32(5), Functor("EqualTo"));
		d_values.to_host(h_values);
		printf("%d %d %d %d %d ", h_values[0], h_values[1], h_values[2], h_values[3], h_values[4]);
		printf("%d %d %d %d %d\n", h_values[5], h_values[6], h_values[7], h_values[8], h_values[9]);
	}

	return 0;
}