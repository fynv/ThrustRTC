#include "JThrustRTC_Native.h"
#include "count.h"
#include "reduce.h"
#include "equal.h"
#include "extrema.h"
#include "inner_product.h"
#include "transform_reduce.h"
#include "logical.h"
#include "partition.h"
#include "sort.h"

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_count(JNIEnv *, jclass, jlong p_vec, jlong p_value)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* value = (DeviceViewable*)(p_value);
	size_t ret;
	if (TRTC_Count(*vec, *value, ret))
		return (int)ret;
	else
		return -1;
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_count_1if(JNIEnv *, jclass, jlong p_vec, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* pred = (Functor*)(p_pred);
	size_t ret;
	if (TRTC_Count_If(*vec, *pred, ret))
		return (int)ret;
	else
		return -1;
}

static jobject s_box_basic_type(JNIEnv *env, const ViewBuf& v, const std::string& type)
{
	if (type == "int8_t")
	{
		jclass classValue = env->FindClass("java/lang/Byte");
		jmethodID valueInit = env->GetMethodID(classValue, "<init>", "(B)V");
		return env->NewObject(classValue, valueInit, *(int8_t*)v.data());
	}
	else if (type == "int16_t")
	{
		jclass classValue = env->FindClass("java/lang/Short");
		jmethodID valueInit = env->GetMethodID(classValue, "<init>", "(S)V");
		return env->NewObject(classValue, valueInit, *(int16_t*)v.data());
	}
	else if (type == "int32_t")
	{
		jclass classValue = env->FindClass("java/lang/Integer");
		jmethodID valueInit = env->GetMethodID(classValue, "<init>", "(I)V");
		return env->NewObject(classValue, valueInit, *(int32_t*)v.data());
	}
	else if (type == "int64_t")
	{
		jclass classValue = env->FindClass("java/lang/Long");
		jmethodID valueInit = env->GetMethodID(classValue, "<init>", "(J)V");
		return env->NewObject(classValue, valueInit, *(int64_t*)v.data());
	}
	else if (type == "float")
	{
		jclass classValue = env->FindClass("java/lang/Float");
		jmethodID valueInit = env->GetMethodID(classValue, "<init>", "(F)V");
		return env->NewObject(classValue, valueInit, *(float*)v.data());
	}
	else if (type == "double")
	{
		jclass classValue = env->FindClass("java/lang/Double");
		jmethodID valueInit = env->GetMethodID(classValue, "<init>", "(D)V");
		return env->NewObject(classValue, valueInit, *(double*)v.data());
	}
	return nullptr;
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_reduce__J(JNIEnv *env, jclass, jlong p_vec)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	ViewBuf ret;
	if (!TRTC_Reduce(*vec, ret)) return nullptr;
	return s_box_basic_type(env, ret, vec->name_elem_cls());
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_reduce__JJ(JNIEnv *env, jclass, jlong p_vec, jlong p_init)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* init = (DeviceViewable*)(p_init);
	ViewBuf ret;
	if (!TRTC_Reduce(*vec, *init, ret)) return nullptr;
	return s_box_basic_type(env, ret, vec->name_elem_cls());
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_reduce__JJJ(JNIEnv *env, jclass, jlong p_vec, jlong p_init, jlong p_binary_op)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	DeviceViewable* init = (DeviceViewable*)(p_init);
	Functor* binary_op = (Functor*)(p_binary_op);
	ViewBuf ret;
	if (!TRTC_Reduce(*vec, *init, *binary_op, ret)) return nullptr;
	return s_box_basic_type(env, ret, vec->name_elem_cls());
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_reduce_1by_1key__JJJJ(JNIEnv *env, jclass, jlong p_key_in, jlong p_value_in, jlong p_key_out, jlong p_value_out)
{
	DVVectorLike* key_in = (DVVectorLike*)(p_key_in);
	DVVectorLike* value_in = (DVVectorLike*)(p_value_in);
	DVVectorLike* key_out = (DVVectorLike*)(p_key_out);
	DVVectorLike* value_out = (DVVectorLike*)(p_value_out);
	return (jint)TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_reduce_1by_1key__JJJJJ(JNIEnv *, jclass, jlong p_key_in, jlong p_value_in, jlong p_key_out, jlong p_value_out, jlong p_binary_pred)
{
	DVVectorLike* key_in = (DVVectorLike*)(p_key_in);
	DVVectorLike* value_in = (DVVectorLike*)(p_value_in);
	DVVectorLike* key_out = (DVVectorLike*)(p_key_out);
	DVVectorLike* value_out = (DVVectorLike*)(p_value_out);
	Functor* binary_pred = (Functor*)(p_binary_pred);
	return (jint)TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out, *binary_pred);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_reduce_1by_1key__JJJJJJ(JNIEnv *, jclass, jlong p_key_in, jlong p_value_in, jlong p_key_out, jlong p_value_out, jlong p_binary_pred, jlong p_binary_op)
{
	DVVectorLike* key_in = (DVVectorLike*)(p_key_in);
	DVVectorLike* value_in = (DVVectorLike*)(p_value_in);
	DVVectorLike* key_out = (DVVectorLike*)(p_key_out);
	DVVectorLike* value_out = (DVVectorLike*)(p_value_out);
	Functor* binary_pred = (Functor*)(p_binary_pred);
	Functor* binary_op = (Functor*)(p_binary_op);
	return TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out, *binary_pred, *binary_op);
}

static jobject s_bool_create(JNIEnv *env, bool b)
{
	jclass classValue = env->FindClass("java/lang/Boolean");
	jmethodID valueInit = env->GetMethodID(classValue, "<init>", "(Z)V");
	return env->NewObject(classValue, valueInit, (jboolean)(b?1:0));
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_equal__JJ(JNIEnv *env, jclass, jlong p_vec1, jlong p_vec2)
{
	DVVectorLike* vec1 = (DVVectorLike*)(p_vec1);
	DVVectorLike* vec2 = (DVVectorLike*)(p_vec2);
	bool ret;
	if (!TRTC_Equal(*vec1, *vec2, ret))
		return nullptr;
	return s_bool_create(env, ret);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_equal__JJJ(JNIEnv *env, jclass, jlong p_vec1, jlong p_vec2, jlong p_binary_pred)
{
	DVVectorLike* vec1 = (DVVectorLike*)(p_vec1);
	DVVectorLike* vec2 = (DVVectorLike*)(p_vec2);
	Functor* binary_pred = (Functor*)(p_binary_pred);
	bool ret;
	if (!TRTC_Equal(*vec1, *vec2, *binary_pred, ret))
		return nullptr;
	return s_bool_create(env, ret);
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_min_1element__J(JNIEnv *, jclass, jlong p_vec)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	size_t id_min;
	if (!TRTC_Min_Element(*vec, id_min)) return -1;
	return (jint)id_min;
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_min_1element__JJ(JNIEnv *, jclass, jlong p_vec, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* comp = (Functor*)(p_comp);
	size_t id_min;
	if (!TRTC_Min_Element(*vec, *comp, id_min))	return -1;
	return (jint)id_min;
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_max_1element__J(JNIEnv *, jclass, jlong p_vec)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	size_t id_max;
	if (!TRTC_Max_Element(*vec, id_max)) return -1;
	return id_max;
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_max_1element__JJ(JNIEnv *, jclass, jlong p_vec, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* comp = (Functor*)(p_comp);
	size_t id_max;
	if (!TRTC_Max_Element(*vec, *comp, id_max))	return -1;
	return id_max;
}

JNIEXPORT jintArray JNICALL Java_JThrustRTC_Native_minmax_1element__J(JNIEnv *env, jclass, jlong p_vec)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	size_t id_min, id_max;
	if (!TRTC_MinMax_Element(*vec, id_min, id_max))	return nullptr;
	jintArray j_ret = env->NewIntArray(2);
	jint* p_ret = env->GetIntArrayElements(j_ret, nullptr);
	p_ret[0] = (jint)id_min;
	p_ret[1] = (jint)id_max;
	env->ReleaseIntArrayElements(j_ret, p_ret, 0);
	return j_ret;
}

JNIEXPORT jintArray JNICALL Java_JThrustRTC_Native_minmax_1element__JJ(JNIEnv *env, jclass, jlong p_vec, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* comp = (Functor*)(p_comp);
	size_t id_min, id_max;
	if (!TRTC_MinMax_Element(*vec, *comp, id_min, id_max)) return nullptr;
	jintArray j_ret = env->NewIntArray(2);
	jint* p_ret = env->GetIntArrayElements(j_ret, nullptr);
	p_ret[0] = (jint)id_min;
	p_ret[1] = (jint)id_max;
	env->ReleaseIntArrayElements(j_ret, p_ret, 0);
	return j_ret;
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_inner_1product__JJJ(JNIEnv *env, jclass, jlong p_vec1, jlong p_vec2, jlong p_init)
{
	DVVectorLike* vec1 = (DVVectorLike*)(p_vec1);
	DVVectorLike* vec2 = (DVVectorLike*)(p_vec2);
	DeviceViewable* init = (DeviceViewable*)(p_init);
	ViewBuf ret;
	if (!TRTC_Inner_Product(*vec1, *vec2, *init, ret))
		return nullptr;
	return s_box_basic_type(env, ret, init->name_view_cls());
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_inner_1product__JJJJJ(JNIEnv *env, jclass, jlong p_vec1, jlong p_vec2, jlong p_init, jlong p_binary_op1, jlong p_binary_op2)
{
	DVVectorLike* vec1 = (DVVectorLike*)(p_vec1);
	DVVectorLike* vec2 = (DVVectorLike*)(p_vec2);
	DeviceViewable* init = (DeviceViewable*)(p_init);
	Functor* binary_op1 = (Functor*)(p_binary_op1);
	Functor* binary_op2 = (Functor*)(p_binary_op2);
	ViewBuf ret;
	if (!TRTC_Inner_Product(*vec1, *vec2, *init, ret, *binary_op1, *binary_op2))
		return nullptr;
	return s_box_basic_type(env, ret, init->name_view_cls());
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_transform_1reduce(JNIEnv *env, jclass, jlong p_vec, jlong p_unary_op, jlong p_init, jlong p_binary_op)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* unary_op = (Functor*)(p_unary_op);
	DeviceViewable* init = (DeviceViewable*)(p_init);
	Functor* binary_op = (Functor*)(p_binary_op);
	ViewBuf ret;
	if (!TRTC_Transform_Reduce(*vec, *unary_op, *init, *binary_op, ret))
		return nullptr;
	return s_box_basic_type(env, ret, init->name_view_cls());
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_all_1of(JNIEnv *env, jclass, jlong p_vec, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* pred = (Functor*)(p_pred);
	bool ret;
	if (!TRTC_All_Of(*vec, *pred, ret))
		return nullptr;
	return s_bool_create(env, ret);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_any_1of(JNIEnv *env, jclass, jlong p_vec, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* pred = (Functor*)(p_pred);
	bool ret;
	if (!TRTC_Any_Of(*vec, *pred, ret))
		return nullptr;
	return s_bool_create(env, ret);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_none_1of(JNIEnv *env, jclass, jlong p_vec, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* pred = (Functor*)(p_pred);
	bool ret;
	if (!TRTC_None_Of(*vec, *pred, ret))
		return nullptr;
	return s_bool_create(env, ret);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_is_1partitioned(JNIEnv *env, jclass, jlong p_vec, jlong p_pred)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* pred = (Functor*)(p_pred);
	bool ret;
	if (!TRTC_Is_Partitioned(*vec, *pred, ret))
		return nullptr;
	return s_bool_create(env, ret);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_is_1sorted__J(JNIEnv *env, jclass, jlong p_vec)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	bool ret;
	if (!TRTC_Is_Sorted(*vec, ret))
		return nullptr;
	return s_bool_create(env, ret);
}

JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_is_1sorted__JJ(JNIEnv *env, jclass, jlong p_vec, jlong p_comp)
{
	DVVectorLike* vec = (DVVectorLike*)(p_vec);
	Functor* comp = (Functor*)(p_comp);
	bool ret;
	if (!TRTC_Is_Sorted(*vec, *comp, ret))
		return nullptr;
	return s_bool_create(env, ret);
}
