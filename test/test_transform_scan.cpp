#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVCounter.h"
#include "transform_scan.h"

int main()
{
	

	{
		int data[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector d_data("int32_t", 6, data);

		TRTC_Transform_Inclusive_Scan(d_data, d_data, Functor("Negate"), Functor("Plus"));

		d_data.to_host(data);
		printf("%d %d %d %d %d %d\n", data[0], data[1], data[2], data[3], data[4], data[5]);
	}

	{
		int data[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector d_data("int32_t", 6, data);

		TRTC_Transform_Exclusive_Scan(d_data, d_data, Functor("Negate"), DVInt32(4), Functor("Plus"));

		d_data.to_host(data);
		printf("%d %d %d %d %d %d\n", data[0], data[1], data[2], data[3], data[4], data[5]);
	}


	return 0;
}
