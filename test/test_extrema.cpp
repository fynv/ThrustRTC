#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "extrema.h"
#include "built_in.h"

int main()
{
	{
		int h_data[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector d_data("int32_t", 6, h_data);
		size_t id_min;
		TRTC_Min_Element(d_data, id_min);
		printf("%zu %d\n", id_min, h_data[id_min]);
	}

	{
		Pair<int, int> h_data[4] = { { 4,5 }, { 0,7 }, { 2,3 }, { 6,1 }};
		DVVector d_data("Pair<int, int>", 4, h_data);

		Functor compare_key_value = { {},{ "lhs", "rhs" }, "        return  lhs.first < rhs.first;\n" };
		size_t id_min;
		TRTC_Min_Element(d_data, compare_key_value, id_min);
		printf("%zu (%d, %d)\n", id_min, h_data[id_min].first, h_data[id_min].second);
	}

	{
		int h_data[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector d_data("int32_t", 6, h_data);
		size_t id_max;
		TRTC_Max_Element(d_data, id_max);
		printf("%zu %d\n", id_max, h_data[id_max]);
	}

	{
		Pair<int, int> h_data[4] = { { 4,5 }, { 0,7 }, { 2,3 }, { 6,1 } };
		DVVector d_data("Pair<int, int>", 4, h_data);

		Functor compare_key_value = { {},{ "lhs", "rhs" }, "        return  lhs.first < rhs.first;\n" };
		size_t id_max;
		TRTC_Max_Element(d_data, compare_key_value, id_max);
		printf("%zu (%d, %d)\n", id_max, h_data[id_max].first, h_data[id_max].second);
	}

	{
		int h_data[6] = { 1, 0, 2, 2, 1, 3 };
		DVVector d_data("int32_t", 6, h_data);
		size_t id_min, id_max;
		TRTC_MinMax_Element(d_data, id_min, id_max);
		printf("%zu %d\n", id_min, h_data[id_min]);
		printf("%zu %d\n", id_max, h_data[id_max]);
	}

	{
		Pair<int, int> h_data[4] = { { 4,5 }, { 0,7 }, { 2,3 }, { 6,1 } };
		DVVector d_data("Pair<int, int>", 4, h_data);

		Functor compare_key_value = { {},{ "lhs", "rhs" }, "        return  lhs.first < rhs.first;\n" };
		size_t id_min, id_max;
		TRTC_MinMax_Element(d_data, compare_key_value, id_min, id_max);
		printf("%zu (%d, %d)\n", id_min, h_data[id_min].first, h_data[id_min].second);
		printf("%zu (%d, %d)\n", id_max, h_data[id_max].first, h_data[id_max].second);

	}

	return 0;
}
