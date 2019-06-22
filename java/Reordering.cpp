#include "JThrustRTC_Native.h"
#include "copy.h"
#include "remove.h"
#include "unique.h"
#include "partition.h"

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_copy_1if(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* pred = (Functor*)(p_pred);
	return (jint)TRTC_Copy_If(*vec_in, *vec_out, *pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_copy_1if_1stencil(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_stencil, jlong p_vec_out, jlong p_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_stencil = (DVVectorLike*)(p_vec_stencil);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* pred = (Functor*)(p_pred);
	return (jint)TRTC_Copy_If_Stencil(*vec_in, *vec_stencil, *vec_out, *pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove(JNIEnv *, jclass, jlong p_vec, jlong p_value)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value = (DVVectorLike*)(p_value);
	return (jint)TRTC_Remove(*vec, *value);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove_1copy(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_value)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	DeviceViewable* value = (DeviceViewable*)(p_value);
	return (jint)TRTC_Remove_Copy(*vec_in, *vec_out, *value);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove_1if(JNIEnv *, jclass, jlong p_vec, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* pred = (Functor*)(p_pred);
	return (jint)TRTC_Remove_If(*vec, *pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove_1copy_1if(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* pred = (Functor*)(p_pred);
	return (jint)TRTC_Remove_Copy_If(*vec_in, *vec_out, *pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove_1if_1stencil(JNIEnv *, jclass, jlong p_vec, jlong p_stencil, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DVVectorLike* stencil = (DVVectorLike*)(p_stencil);
	Functor* pred = (Functor*)(p_pred);
	return (jint)TRTC_Remove_If_Stencil(*vec, *stencil, *pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove_1copy_1if_1stencil(JNIEnv *, jclass, jlong p_vec_in, jlong p_stencil, jlong p_vec_out, jlong p_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* stencil = (DVVectorLike*)(p_stencil);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* pred = (Functor*)(p_pred);
	return (jint)TRTC_Remove_Copy_If_Stencil(*vec_in, *stencil, *vec_out, *pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique__J(JNIEnv *, jclass, jlong p_vec)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	return (jint)TRTC_Unique(*vec);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique__JJ(JNIEnv *, jclass, jlong p_vec, jlong p_binary_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* binary_pred = (Functor*)(p_binary_pred);
	return (jint)TRTC_Unique(*vec, *binary_pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1copy__JJ(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return (jint)TRTC_Unique_Copy(*vec_in, *vec_out);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1copy__JJJ(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_binary_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* binary_pred = (Functor*)(p_binary_pred);
	return (jint)TRTC_Unique_Copy(*vec_in, *vec_out, *binary_pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1by_1key__JJ(JNIEnv *, jclass, jlong p_keys, jlong p_values)
{
	DVVectorLike* keys = (DVVectorLike*)(p_keys);
	DVVectorLike* values = (DVVectorLike*)(p_values);
	return (jint)TRTC_Unique_By_Key(*keys, *values);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1by_1key__JJJ(JNIEnv *, jclass, jlong p_keys, jlong p_values, jlong p_binary_pred)
{
	DVVectorLike* keys = (DVVectorLike*)(p_keys);
	DVVectorLike* values = (DVVectorLike*)(p_values);
	Functor* binary_pred = (Functor*)(p_binary_pred);
	return (jint)TRTC_Unique_By_Key(*keys, *values, *binary_pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1by_1key_1copy__JJJJ(JNIEnv *, jclass, jlong p_keys_in, jlong p_values_in, jlong p_keys_out, jlong p_values_out)
{
	DVVectorLike* keys_in = (DVVectorLike*)(p_keys_in);
	DVVectorLike* values_in = (DVVectorLike*)(p_values_in);
	DVVectorLike* keys_out = (DVVectorLike*)(p_keys_out);
	DVVectorLike* values_out = (DVVectorLike*)(p_values_out);
	return (jint)TRTC_Unique_By_Key_Copy(*keys_in, *values_in, *keys_out, *values_out);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1by_1key_1copy__JJJJJ(JNIEnv *, jclass, jlong p_keys_in, jlong p_values_in, jlong p_keys_out, jlong p_values_out, jlong p_binary_pred)
{
	DVVectorLike* keys_in = (DVVectorLike*)(p_keys_in);
	DVVectorLike* values_in = (DVVectorLike*)(p_values_in);
	DVVectorLike* keys_out = (DVVectorLike*)(p_keys_out);
	DVVectorLike* values_out = (DVVectorLike*)(p_values_out);
	Functor* binary_pred = (Functor*)(p_binary_pred);
	return (jint)TRTC_Unique_By_Key_Copy(*keys_in, *values_in, *keys_out, *values_out, *binary_pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_partition(JNIEnv *, jclass, jlong p_vec, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* pred = (Functor*)(p_pred);
	return (jint)TRTC_Partition(*vec, *pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_partition_1stencil(JNIEnv *, jclass, jlong p_vec, jlong p_stencil, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DVVectorLike* stencil = (DVVectorLike*)(p_stencil);
	Functor* pred = (Functor*)(p_pred);
	return (jint)TRTC_Partition_Stencil(*vec, *stencil, *pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_partition_1copy(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_true, jlong p_vec_false, jlong p_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_true = (DVVectorLike*)(p_vec_true);
	DVVectorLike* vec_false = (DVVectorLike*)(p_vec_false);
	Functor* pred = (Functor*)(p_pred);
	return (jint)TRTC_Partition_Copy(*vec_in, *vec_true, *vec_false, *pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_partition_1copy_1stencil(JNIEnv *, jclass, jlong p_vec_in, jlong p_stencil, jlong p_vec_true, jlong p_vec_false, jlong p_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_stencil = (DVVectorLike*)(p_stencil);
	DVVectorLike* vec_true = (DVVectorLike*)(p_vec_true);
	DVVectorLike* vec_false = (DVVectorLike*)(p_vec_false);
	Functor* pred = (Functor*)(p_pred);
	return TRTC_Partition_Copy_Stencil(*vec_in, *vec_stencil, *vec_true, *vec_false, *pred);
}
