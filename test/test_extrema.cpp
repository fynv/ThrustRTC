#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "extrema.h"

int main()
{
	TRTCContext::set_ptx_cache("__ptx_cache__");
	TRTCContext ctx;

	{
		int h_data[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector d_data(ctx, "int32_t", 6, h_data);
		size_t id_min;
		TRTC_Min_Element(ctx, d_data, id_min);
		printf("%lu %d\n", id_min, h_data[id_min]);
	}

	{

		struct key_value
		{
			int key;
			int value;
		};

		std::string d_key_value = ctx.add_struct(
			"    int key;\n"
			"    int value;\n"
		);

		key_value h_data[4] = { { 4,5 }, { 0,7 }, { 2,3 }, { 6,1 }};
		DVVector d_data(ctx, d_key_value.c_str(), 4, h_data);

		Functor compare_key_value = { {},{ "lhs", "rhs" }, "ret", "        ret =  lhs.key < rhs.key;\n" };
		size_t id_min;
		TRTC_Min_Element(ctx, d_data, compare_key_value, id_min);
		printf("%lu (%d, %d)\n", id_min, h_data[id_min].key, h_data[id_min].value);

	}

	{
		int h_data[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector d_data(ctx, "int32_t", 6, h_data);
		size_t id_max;
		TRTC_Max_Element(ctx, d_data, id_max);
		printf("%lu %d\n", id_max, h_data[id_max]);
	}

	{

		struct key_value
		{
			int key;
			int value;
		};

		std::string d_key_value = ctx.add_struct(
			"    int key;\n"
			"    int value;\n"
		);

		key_value h_data[4] = { { 4,5 }, { 0,7 }, { 2,3 }, { 6,1 } };
		DVVector d_data(ctx, d_key_value.c_str(), 4, h_data);

		Functor compare_key_value = { {},{ "lhs", "rhs" }, "ret", "        ret =  lhs.key < rhs.key;\n" };
		size_t id_max;
		TRTC_Max_Element(ctx, d_data, compare_key_value, id_max);
		printf("%lu (%d, %d)\n", id_max, h_data[id_max].key, h_data[id_max].value);

	}

	{
		int h_data[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector d_data(ctx, "int32_t", 6, h_data);
		size_t id_min, id_max;
		TRTC_MinMax_Element(ctx, d_data, id_min, id_max);
		printf("%lu %d\n", id_min, h_data[id_min]);
		printf("%lu %d\n", id_max, h_data[id_max]);
	}

	{

		struct key_value
		{
			int key;
			int value;
		};

		std::string d_key_value = ctx.add_struct(
			"    int key;\n"
			"    int value;\n"
		);

		key_value h_data[4] = { { 4,5 }, { 0,7 }, { 2,3 }, { 6,1 } };
		DVVector d_data(ctx, d_key_value.c_str(), 4, h_data);

		Functor compare_key_value = { {},{ "lhs", "rhs" }, "ret", "        ret =  lhs.key < rhs.key;\n" };
		size_t id_min, id_max;
		TRTC_MinMax_Element(ctx, d_data, compare_key_value, id_min, id_max);
		printf("%lu (%d, %d)\n", id_min, h_data[id_min].key, h_data[id_min].value);
		printf("%lu (%d, %d)\n", id_max, h_data[id_max].key, h_data[id_max].value);

	}

	return 0;
}
