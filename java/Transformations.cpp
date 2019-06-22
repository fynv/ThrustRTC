#include "JThrustRTC_Native.h"
#include "fill.h"
#include "replace.h"
#include "for_each.h"
#include "adjacent_difference.h"
#include "sequence.h"
#include "tabulate.h"
#include "transform.h"

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_fill(JNIEnv *, jclass, jlong p_vec, jlong p_value)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value = (DeviceViewable*)(p_value);
	return TRTC_Fill(*vec, *value) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_replace(JNIEnv *, jclass, jlong p_vec, jlong p_old_value, jlong p_new_value)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* old_value = (DeviceViewable*)(p_old_value);
	DeviceViewable* new_value = (DeviceViewable*)(p_new_value);
	return TRTC_Replace(*vec, *old_value, *new_value) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_replace_1if(JNIEnv *, jclass, jlong p_vec, jlong p_pred, jlong p_new_value)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* pred = (Functor*)(p_pred);
	DeviceViewable* new_value = (DeviceViewable*)(p_new_value);
	return TRTC_Replace_If(*vec, *pred, *new_value) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_replace_1copy(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_old_value, jlong p_new_value)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	DeviceViewable* old_value = (DeviceViewable*)(p_old_value);
	DeviceViewable* new_value = (DeviceViewable*)(p_new_value);
	return TRTC_Replace_Copy(*vec_in, *vec_out, *old_value, *new_value) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_replace_1copy_1if(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_pred, jlong p_new_value)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* pred = (Functor*)(p_pred);
	DeviceViewable* new_value = (DeviceViewable*)(p_new_value);
	return TRTC_Replace_Copy_If(*vec_in, *vec_out, *pred, *new_value) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_for_1each(JNIEnv *, jclass, jlong p_vec, jlong p_f)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* f = (Functor*)(p_f);
	return TRTC_For_Each(*vec, *f) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_adjacent_1difference__JJ(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return TRTC_Adjacent_Difference(*vec_in, *vec_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_adjacent_1difference__JJJ(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_binary_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* binary_op = (Functor*)(p_binary_op);
	return TRTC_Adjacent_Difference(*vec_in, *vec_out, *binary_op) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sequence__J(JNIEnv *, jclass, jlong p_vec)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	return TRTC_Sequence(*vec) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sequence__JJ(JNIEnv *, jclass, jlong p_vec, jlong p_value_init)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value_init = (DeviceViewable*)(p_value_init);
	return TRTC_Sequence(*vec, *value_init) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sequence__JJJ(JNIEnv *, jclass, jlong p_vec, jlong p_value_init, jlong p_value_step)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value_init = (DeviceViewable*)(p_value_init);
	DeviceViewable* value_step = (DeviceViewable*)(p_value_step);
	return TRTC_Sequence(*vec, *value_init, *value_step) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_tabulate(JNIEnv *, jclass, jlong p_vec, jlong p_op)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* op = (Functor*)(p_op);
	return TRTC_Tabulate(*vec, *op) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* op = (Functor*)(p_op);
	return TRTC_Transform(*vec_in, *vec_out, *op) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1binary(JNIEnv *, jclass, jlong p_vec_in1, jlong p_vec_in2, jlong p_vec_out, jlong p_op)
{
	DVVectorLike* vec_in1 = (DVVectorLike*)(p_vec_in1);
	DVVectorLike* vec_in2 = (DVVectorLike*)(p_vec_in2);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* op = (Functor*)(p_op);
	return TRTC_Transform_Binary(*vec_in1, *vec_in2, *vec_out, *op) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1if(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out, jlong p_op, jlong p_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* op = (Functor*)(p_op);
	Functor* pred = (Functor*)(p_pred);
	return TRTC_Transform_If(*vec_in, *vec_out, *op, *pred) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1if_1stencil(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_stencil, jlong p_vec_out, jlong p_op, jlong p_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_stencil = (DVVectorLike*)(p_vec_stencil);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* op = (Functor*)(p_op);
	Functor* pred = (Functor*)(p_pred);
	return TRTC_Transform_If_Stencil(*vec_in, *vec_stencil, *vec_out, *op, *pred) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1binary_1if_1stencil(JNIEnv *, jclass, jlong p_vec_in1, jlong p_vec_in2, jlong p_vec_stencil, jlong p_vec_out, jlong p_op, jlong p_pred)
{
	DVVectorLike* vec_in1 = (DVVectorLike*)(p_vec_in1);
	DVVectorLike* vec_in2 = (DVVectorLike*)(p_vec_in2);
	DVVectorLike* vec_stencil = (DVVectorLike*)(p_vec_stencil);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* op = (Functor*)(p_op);
	Functor* pred = (Functor*)(p_pred);
	return TRTC_Transform_Binary_If_Stencil(*vec_in1, *vec_in2, *vec_stencil, *vec_out, *op, *pred) ? 1 : 0;
}
