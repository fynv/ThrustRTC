#include "JThrustRTC_Native.h"
#include "scan.h"
#include "transform_scan.h"

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_inclusive_1scan__JJ(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return TRTC_Inclusive_Scan(*vec_in, *vec_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_inclusive_1scan__JJJ(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_binary_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* binary_op = (Functor*)(p_binary_op);
	return TRTC_Inclusive_Scan(*vec_in, *vec_out, *binary_op) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan__JJ(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return TRTC_Exclusive_Scan(*vec_in, *vec_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan__JJJ(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_init)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	DeviceViewable* init = (DeviceViewable*)(p_init);
	return TRTC_Exclusive_Scan(*vec_in, *vec_out, *init) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan__JJJJ(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_init, jlong p_binary_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	DeviceViewable* init = (DeviceViewable*)(p_init);
	Functor* binary_op = (Functor*)(p_binary_op);
	return TRTC_Exclusive_Scan(*vec_in, *vec_out, *init, *binary_op) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_inclusive_1scan_1by_1key__JJJ(JNIEnv *, jclass, jlong p_vec_key, jlong p_vec_value, jlong p_vec_out)
{
	DVVectorLike* vec_key = (DVVectorLike*)(p_vec_key);
	DVVectorLike* vec_value = (DVVectorLike*)(p_vec_value);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_inclusive_1scan_1by_1key__JJJJ(JNIEnv *, jclass, jlong p_vec_key, jlong p_vec_value, jlong p_vec_out, jlong p_binary_pred)
{
	DVVectorLike* vec_key = (DVVectorLike*)(p_vec_key);
	DVVectorLike* vec_value = (DVVectorLike*)(p_vec_value);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* binary_pred = (Functor*)(p_binary_pred);
	return TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *binary_pred) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_inclusive_1scan_1by_1key__JJJJJ(JNIEnv *, jclass, jlong p_vec_key, jlong p_vec_value, jlong p_vec_out, jlong p_binary_pred, jlong p_binary_op)
{
	DVVectorLike* vec_key = (DVVectorLike*)(p_vec_key);
	DVVectorLike* vec_value = (DVVectorLike*)(p_vec_value);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* binary_pred = (Functor*)(p_binary_pred);
	Functor* binary_op = (Functor*)(p_binary_op);
	return TRTC_Inclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *binary_pred, *binary_op) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan_1by_1key__JJJ(JNIEnv *, jclass, jlong p_vec_key, jlong p_vec_value, jlong p_vec_out)
{
	DVVectorLike* vec_key = (DVVectorLike*)(p_vec_key);
	DVVectorLike* vec_value = (DVVectorLike*)(p_vec_value);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan_1by_1key__JJJJ(JNIEnv *, jclass, jlong p_vec_key, jlong p_vec_value, jlong p_vec_out, jlong p_init)
{
	DVVectorLike* vec_key = (DVVectorLike*)(p_vec_key);
	DVVectorLike* vec_value = (DVVectorLike*)(p_vec_value);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	DeviceViewable* init = (DeviceViewable*)(p_init);
	return TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan_1by_1key__JJJJJ(JNIEnv *, jclass, jlong p_vec_key, jlong p_vec_value, jlong p_vec_out, jlong p_init, jlong p_binary_pred)
{
	DVVectorLike* vec_key = (DVVectorLike*)(p_vec_key);
	DVVectorLike* vec_value = (DVVectorLike*)(p_vec_value);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	DeviceViewable* init = (DeviceViewable*)(p_init);
	Functor* binary_pred = (Functor*)(p_binary_pred);
	return TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init, *binary_pred) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan_1by_1key__JJJJJJ(JNIEnv *, jclass, jlong p_vec_key, jlong p_vec_value, jlong p_vec_out, jlong p_init, jlong p_binary_pred, jlong p_binary_op)
{
	DVVectorLike* vec_key = (DVVectorLike*)(p_vec_key);
	DVVectorLike* vec_value = (DVVectorLike*)(p_vec_value);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	DeviceViewable* init = (DeviceViewable*)(p_init);
	Functor* binary_pred = (Functor*)(p_binary_pred);
	Functor* binary_op = (Functor*)(p_binary_op);
	return TRTC_Exclusive_Scan_By_Key(*vec_key, *vec_value, *vec_out, *init, *binary_pred, *binary_op) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1inclusive_1scan(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_unary_op, jlong p_binary_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* unary_op = (Functor*)(p_unary_op);
	Functor* binary_op = (Functor*)(p_binary_op);
	return TRTC_Transform_Inclusive_Scan(*vec_in, *vec_out, *unary_op, *binary_op) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1exclusive_1scan(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_unary_op, jlong p_init, jlong p_binary_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* unary_op = (Functor*)(p_unary_op);
	DeviceViewable* init = (DeviceViewable*)(p_init);
	Functor* binary_op = (Functor*)(p_binary_op);
	return TRTC_Transform_Exclusive_Scan(*vec_in, *vec_out, *unary_op, *init, *binary_op) ? 1 : 0;
}