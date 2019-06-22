#include "JThrustRTC_Native.h"
#include "sort.h"

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sort__J(JNIEnv *, jclass, jlong p_vec)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	return TRTC_Sort(*vec) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sort__JJ(JNIEnv *, jclass, jlong p_vec, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* comp = (Functor*)(p_comp);
	return TRTC_Sort(*vec, *comp) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sort_1by_1key__JJ(JNIEnv *, jclass, jlong p_keys, jlong p_values)
{
	DVVectorLike* keys = (DVVectorLike*)(p_keys);
	DVVectorLike* values = (DVVectorLike*)(p_values);
	return TRTC_Sort_By_Key(*keys, *values) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sort_1by_1key__JJJ(JNIEnv *, jclass, jlong p_keys, jlong p_values, jlong p_comp)
{
	DVVectorLike* keys = (DVVectorLike*)(p_keys);
	DVVectorLike* values = (DVVectorLike*)(p_values);
	Functor* comp = (Functor*)(p_comp);
	return TRTC_Sort_By_Key(*keys, *values, *comp) ? 1 : 0;
}
