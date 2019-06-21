#include "JThrustRTC_Native.h"
#include "fill.h"

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_fill(JNIEnv *, jclass, jlong p_vec, jlong p_value)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value =(DeviceViewable*)(p_value);
	return TRTC_Fill(*vec, *value) ? 1 : 0;
}
