#include "JThrustRTC_Native.h"
#include "merge.h"

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_merge__JJJ(JNIEnv *, jclass, jlong p_vec1, jlong p_vec2, jlong p_vec_out)
{
	DVVectorLike* vec1 = (DVVectorLike*)(p_vec1);
	DVVectorLike* vec2 = (DVVectorLike*)(p_vec2);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return TRTC_Merge(*vec1, *vec2, *vec_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_merge__JJJJ(JNIEnv *, jclass, jlong p_vec1, jlong p_vec2, jlong p_vec_out, jlong p_comp)
{
	DVVectorLike* vec1 = (DVVectorLike*)(p_vec1);
	DVVectorLike* vec2 = (DVVectorLike*)(p_vec2);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* comp = (Functor*)(p_comp);
	return TRTC_Merge(*vec1, *vec2, *vec_out, *comp) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_merge_1by_1key__JJJJJJ(JNIEnv *, jclass, jlong p_keys1, jlong p_keys2, jlong p_value1, jlong p_value2, jlong p_keys_out, jlong p_value_out)
{
	DVVectorLike* keys1 = (DVVectorLike*)(p_keys1);
	DVVectorLike* keys2 = (DVVectorLike*)(p_keys2);
	DVVectorLike* value1 = (DVVectorLike*)(p_value1);
	DVVectorLike* value2 = (DVVectorLike*)(p_value2);
	DVVectorLike* keys_out = (DVVectorLike*)(p_keys_out);
	DVVectorLike* value_out = (DVVectorLike*)(p_value_out);
	return TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_merge_1by_1key__JJJJJJJ(JNIEnv *, jclass, jlong p_keys1, jlong p_keys2, jlong p_value1, jlong p_value2, jlong p_keys_out, jlong p_value_out, jlong p_comp)
{
	DVVectorLike* keys1 = (DVVectorLike*)(p_keys1);
	DVVectorLike* keys2 = (DVVectorLike*)(p_keys2);
	DVVectorLike* value1 = (DVVectorLike*)(p_value1);
	DVVectorLike* value2 = (DVVectorLike*)(p_value2);
	DVVectorLike* keys_out = (DVVectorLike*)(p_keys_out);
	DVVectorLike* value_out = (DVVectorLike*)(p_value_out);
	Functor* comp = (Functor*)(p_comp);
	return TRTC_Merge_By_Key(*keys1, *keys2, *value1, *value2, *keys_out, *value_out, *comp) ? 1 : 0;
}