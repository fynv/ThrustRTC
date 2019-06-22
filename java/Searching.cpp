#include "JThrustRTC_Native.h"
#include "find.h"
#include "mismatch.h"
#include "binary_search.h"
#include "partition.h"
#include "sort.h"

static jobject s_int_create(JNIEnv *env, int i)
{
	jclass classValue = env->FindClass("java/lang/Integer");
	jmethodID valueInit = env->GetMethodID(classValue, "<init>", "(I)V");
	return env->NewObject(classValue, valueInit, (jint)i);
}

static jobject s_bool_create(JNIEnv *env, bool b)
{
	jclass classValue = env->FindClass("java/lang/Boolean");
	jmethodID valueInit = env->GetMethodID(classValue, "<init>", "(Z)V");
	return env->NewObject(classValue, valueInit, (jboolean)(b ? 1 : 0));
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_find(JNIEnv *env, jclass, jlong p_vec, jlong p_value)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value = (DeviceViewable*)(p_value);
	size_t result;
	if (!TRTC_Find(*vec, *value, result))
		return nullptr;
	return s_int_create(env, (int)result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_find_1if(JNIEnv *env, jclass, jlong p_vec, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* pred = (Functor*)(p_pred);
	size_t result;
	if (!TRTC_Find_If(*vec, *pred, result))
		return nullptr;
	return s_int_create(env, (int)result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_find_1if_1not(JNIEnv *env, jclass, jlong p_vec, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* pred = (Functor*)(p_pred);
	size_t result;
	if (!TRTC_Find_If_Not(*vec, *pred, result))
		return nullptr;
	return s_int_create(env, (int)result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_mismatch__JJ(JNIEnv *env, jclass, jlong p_vec1, jlong p_vec2)
{
	DVVectorLike* vec1 = (DVVectorLike*)(p_vec1);
	DVVectorLike* vec2 = (DVVectorLike*)(p_vec2);
	size_t result;
	if (!TRTC_Mismatch(*vec1, *vec2, result))
		return nullptr;
	return s_int_create(env, (int)result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_mismatch__JJJ(JNIEnv *env, jclass, jlong p_vec1, jlong p_vec2, jlong p_pred)
{
	DVVectorLike* vec1 = (DVVectorLike*)(p_vec1);
	DVVectorLike* vec2 = (DVVectorLike*)(p_vec2);
	Functor* pred = (Functor*)(p_pred);
	size_t result;
	if (!TRTC_Mismatch(*vec1, *vec2, *pred, result))
		return nullptr;
	return s_int_create(env, (int)result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_lower_1bound__JJ(JNIEnv *env, jclass, jlong p_vec, jlong p_value)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value = (DeviceViewable*)(p_value);
	size_t result;
	if (!TRTC_Lower_Bound(*vec, *value, result))
		return nullptr;
	return s_int_create(env, (int)result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_lower_1bound__JJJ(JNIEnv *env, jclass, jlong p_vec, jlong p_value, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value = (DeviceViewable*)(p_value);
	Functor* comp = (Functor*)(p_comp);
	size_t result;
	if (!TRTC_Lower_Bound(*vec, *value, *comp, result))
		return nullptr;
	return s_int_create(env, (int)result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_upper_1bound__JJ(JNIEnv *env, jclass, jlong p_vec, jlong p_value)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value = (DeviceViewable*)(p_value);
	size_t result;
	if (!TRTC_Upper_Bound(*vec, *value, result))
		return nullptr;
	return s_int_create(env, (int)result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_upper_1bound__JJJ(JNIEnv *env, jclass, jlong p_vec, jlong p_value, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value = (DeviceViewable*)(p_value);
	Functor* comp = (Functor*)(p_comp);
	size_t result;
	if (!TRTC_Upper_Bound(*vec, *value, *comp, result))
		return nullptr;
	return s_int_create(env, (int)result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_binary_1search__JJ(JNIEnv *env, jclass, jlong p_vec, jlong p_value)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value = (DeviceViewable*)(p_value);
	bool result;
	if (!TRTC_Binary_Search(*vec, *value, result))
		return nullptr;
	return s_bool_create(env, result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_binary_1search__JJJ(JNIEnv *env, jclass, jlong p_vec, jlong p_value, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value = (DeviceViewable*)(p_value);
	Functor* comp = (Functor*)(p_comp);
	bool result;
	if (!TRTC_Binary_Search(*vec, *value, *comp, result))
		return nullptr;
	return s_bool_create(env, result);
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_lower_1bound_1v__JJJ(JNIEnv *, jclass, jlong p_vec, jlong p_values, jlong p_result)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DVVectorLike* value = (DVVectorLike*)(p_values);
	DVVectorLike* result = (DVVectorLike*)(p_result);
	return TRTC_Lower_Bound_V(*vec, *value, *result) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_lower_1bound_1v__JJJJ(JNIEnv *, jclass, jlong p_vec, jlong p_values, jlong p_result, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DVVectorLike* value = (DVVectorLike*)(p_values);
	DVVectorLike* result = (DVVectorLike*)(p_result);
	Functor* comp = (Functor*)(p_comp);
	return TRTC_Lower_Bound_V(*vec, *value, *result, *comp) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_upper_1bound_1v__JJJ(JNIEnv *, jclass, jlong p_vec, jlong p_values, jlong p_result)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DVVectorLike* value = (DVVectorLike*)(p_values);
	DVVectorLike* result = (DVVectorLike*)(p_result);
	return TRTC_Upper_Bound_V(*vec, *value, *result) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_upper_1bound_1v__JJJJ(JNIEnv *, jclass, jlong p_vec, jlong p_values, jlong p_result, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DVVectorLike* value = (DVVectorLike*)(p_values);
	DVVectorLike* result = (DVVectorLike*)(p_result);
	Functor* comp = (Functor*)(p_comp);
	return TRTC_Upper_Bound_V(*vec, *value, *result, *comp) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_binary_1search_1v__JJJ(JNIEnv *, jclass, jlong p_vec, jlong p_values, jlong p_result)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DVVectorLike* value = (DVVectorLike*)(p_values);
	DVVectorLike* result = (DVVectorLike*)(p_result);
	return TRTC_Binary_Search_V(*vec, *value, *result) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_binary_1search_1v__JJJJ(JNIEnv *, jclass, jlong p_vec, jlong p_values, jlong p_result, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DVVectorLike* value = (DVVectorLike*)(p_values);
	DVVectorLike* result = (DVVectorLike*)(p_result);
	Functor* comp = (Functor*)(p_comp);
	return TRTC_Binary_Search_V(*vec, *value, *result, *comp) ? 1 : 0;
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_partition_1point(JNIEnv *env, jclass, jlong p_vec, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* pred = (Functor*)(p_pred);
	size_t result;
	if (!TRTC_Partition_Point(*vec, *pred, result))
		return nullptr;
	return s_int_create(env, (int)result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_is_1sorted_1until__J(JNIEnv *env, jclass, jlong p_vec)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	size_t result;
	if (!TRTC_Is_Sorted_Until(*vec, result))
		return nullptr;
	return s_int_create(env, (int)result);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_is_1sorted_1until__JJ(JNIEnv *env, jclass, jlong p_vec, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* comp = (Functor*)(p_comp);
	size_t result;
	if (!TRTC_Is_Sorted_Until(*vec, *comp, result))
		return nullptr;
	return s_int_create(env, (int)result);
}
