#include <stdio.h>
#include <stdlib.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "sort.h"

int main()
{
	
	
	{
		int hvalues[6]= { 1, 4, 2, 8, 5, 7 };
		DVVector dvalues("int32_t", 6, hvalues);
		
		bool res;
		TRTC_Is_Sorted(dvalues, res);
		puts(res ? "true" : "false");

		TRTC_Sort(dvalues);
		dvalues.to_host(hvalues);
		printf("%d %d %d %d %d %d\n", hvalues[0], hvalues[1], hvalues[2], hvalues[3], hvalues[4], hvalues[5]);

		TRTC_Is_Sorted(dvalues, res);
		puts(res ? "true" : "false");
	}

	{
		int hvalues[6] = { 1, 4, 2, 8, 5, 7 };
		DVVector dvalues("int32_t", 6, hvalues);

		bool res;
		TRTC_Is_Sorted(dvalues, Functor("Greater"), res);
		puts(res ? "true" : "false");

		TRTC_Sort(dvalues, Functor("Greater"));
		dvalues.to_host(hvalues);
		printf("%d %d %d %d %d %d\n", hvalues[0], hvalues[1], hvalues[2], hvalues[3], hvalues[4], hvalues[5]);

		TRTC_Is_Sorted(dvalues, Functor("Greater"), res);
		puts(res ? "true" : "false");
	}

	{
		int hkeys[6] = { 1, 4, 2, 8, 5, 7 };
		DVVector dkeys("int32_t", 6, hkeys);
		char hvalues[6] = { 'a', 'b', 'c', 'd', 'e', 'f' };
		DVVector dvalues("int8_t", 6, hvalues);
		TRTC_Sort_By_Key(dkeys, dvalues);
		dkeys.to_host(hkeys);
		dvalues.to_host(hvalues);
		printf("%d %d %d %d %d %d\n", hkeys[0], hkeys[1], hkeys[2], hkeys[3], hkeys[4], hkeys[5]);
		printf("%c %c %c %c %c %c\n", hvalues[0], hvalues[1], hvalues[2], hvalues[3], hvalues[4], hvalues[5]);
	}

	{
		int hkeys[6] = { 1, 4, 2, 8, 5, 7 };
		DVVector dkeys("int32_t", 6, hkeys);
		char hvalues[6] = { 'a', 'b', 'c', 'd', 'e', 'f' };
		DVVector dvalues("int8_t", 6, hvalues);
		TRTC_Sort_By_Key(dkeys, dvalues, Functor("Greater"));
		dkeys.to_host(hkeys);
		dvalues.to_host(hvalues);
		printf("%d %d %d %d %d %d\n", hkeys[0], hkeys[1], hkeys[2], hkeys[3], hkeys[4], hkeys[5]);
		printf("%c %c %c %c %c %c\n", hvalues[0], hvalues[1], hvalues[2], hvalues[3], hvalues[4], hvalues[5]);
	}

	{
		int hvalues[8] = { 0, 1, 2, 3, 0, 1, 2, 3 };
		DVVector dvalues("int32_t", 8, hvalues);
		size_t res;
		TRTC_Is_Sorted_Until(dvalues, res);
		printf("%zu\n", res);
	}

	{
		int hvalues[8] = { 3, 2, 1, 0, 3, 2, 1, 0 };
		DVVector dvalues("int32_t", 8, hvalues);
		size_t res;
		TRTC_Is_Sorted_Until(dvalues, Functor("Greater"), res);
		printf("%zu\n", res);
	}

	// big case
	{
		int hvalues[10000];
		// fill with sequence
		for (int i = 0; i < 10000; i++) hvalues[i] = i + 1;
		// randomize
		for (int i = 0; i < 9999; i++)
		{
			int j = i + rand() % (10000 - i);
			if (j != i)
			{
				int tmp = hvalues[i];
				hvalues[i] = hvalues[j];
				hvalues[j] = tmp;
			}
		}
		{
			FILE *fp = fopen("before_sort1.txt", "w");
			for (int i = 0; i < 10000; i++)
				fprintf(fp, "%d\n", hvalues[i]);
			fclose(fp);
		}

		DVVector dvalues("int32_t", 10000, hvalues);
		TRTC_Sort(dvalues);
		dvalues.to_host(hvalues);
		{
			FILE *fp = fopen("after_sort1.txt", "w");
			for (int i = 0; i < 10000; i++)
			{
				if (hvalues[i] != i + 1)
					printf("error: %d %d\n", i + 1, hvalues[i]);
				fprintf(fp, "%d\n", hvalues[i]);
			}
			fclose(fp);
		}

	}

	// big case
	{
		int hkeys[10000];
		int hvalues[10000];
		for (int i = 0; i < 10000; i++)
		{
			hkeys[i] = i + 1;
			hvalues[i] = 10000 - i;
		}
		// randomize
		for (int i = 0; i < 9999; i++)
		{
			int j = i + rand() % (10000 - i);
			if (j != i)
			{
				int tmp = hkeys[i];
				hkeys[i] = hkeys[j];
				hkeys[j] = tmp;

				tmp = hvalues[i];
				hvalues[i] = hvalues[j];
				hvalues[j] = tmp;
			}
		}
		{
			FILE *fp = fopen("before_sort2.txt", "w");
			for (int i = 0; i < 10000; i++)
				fprintf(fp, "%d %d\n", hkeys[i], hvalues[i]);
			fclose(fp);
		}

		DVVector dkeys("int32_t", 10000, hkeys);
		DVVector dvalues("int32_t", 10000, hvalues);
		TRTC_Sort_By_Key(dkeys, dvalues);
		dkeys.to_host(hkeys);
		dvalues.to_host(hvalues);
		{
			FILE *fp = fopen("after_sort2.txt", "w");
			for (int i = 0; i < 10000; i++)
			{
				if (hkeys[i] != i + 1 || hvalues[i]!=10000-i)
					printf("error: %d %d %d\n", i + 1, hkeys[i], hvalues[i]);
				fprintf(fp, "%d %d\n", hkeys[i], hvalues[i]);
			}
			fclose(fp);
		}

	}
	return 0;
}
