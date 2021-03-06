#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"
#include "replace.h"

int main()
{
	Functor is_less_than_zero = { {}, { "x" }, "        return x<0;\n" };
	
	// replace
	int hvec1[5] = { 1,2,3,1,2 };
	DVVector vec1("int32_t", 5, hvec1);
	TRTC_Replace(vec1, DVInt32(1), DVInt32(99));
	vec1.to_host(hvec1);
	printf("%d %d %d %d %d\n", hvec1[0], hvec1[1], hvec1[2], hvec1[3], hvec1[4]);

	// replace_if
	int hvec2[5] = { 1, -2, 3, -4, 5 };
	DVVector vec2("int32_t", 5, hvec2);
	TRTC_Replace_If(vec2, is_less_than_zero, DVInt32(0));
	vec2.to_host(hvec2);
	printf("%d %d %d %d %d\n", hvec2[0], hvec2[1], hvec2[2], hvec2[3], hvec2[4]);

	// replace_copy
	int hvec3[5] = { 1, 2, 3, 1, 2 };
	DVVector vec3_in("int32_t", 5, hvec3);
	DVVector vec3_out("int32_t", 5);
	TRTC_Replace_Copy(vec3_in, vec3_out, DVInt32(1), DVInt32(99));
	vec3_out.to_host(hvec3);
	printf("%d %d %d %d %d\n", hvec3[0], hvec3[1], hvec3[2], hvec3[3], hvec3[4]);

	// replace_copy_if
	int hvec4[5] = { 1, -2, 3, -4, 5 };
	DVVector vec4_in("int32_t", 5, hvec4);
	DVVector vec4_out("int32_t", 5);
	TRTC_Replace_Copy_If(vec4_in, vec4_out, is_less_than_zero, DVInt32(0));
	vec4_out.to_host(hvec4);
	printf("%d %d %d %d %d\n", hvec4[0], hvec4[1], hvec4[2], hvec4[3], hvec4[4]);

	return 0;
}
