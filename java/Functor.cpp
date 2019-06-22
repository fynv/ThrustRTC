#include "JThrustRTC_Native.h"
#include "functor.h"

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_functor_1create___3Ljava_lang_String_2Ljava_lang_String_2(JNIEnv * env, jclass, jobjectArray j_functor_params, jstring j_code_body)
{
	jsize num_functor_params = env->GetArrayLength(j_functor_params);
	std::vector<const char*> functor_params(num_functor_params);
	for (int i = 0; i < num_functor_params; i++)
		functor_params[i] = env->GetStringUTFChars((jstring)env->GetObjectArrayElement(j_functor_params, i), nullptr);
	const char* code_body = env->GetStringUTFChars(j_code_body, nullptr);
	std::vector<CapturedDeviceViewable> arg_map(0);
	Functor* cptr = new Functor(arg_map, functor_params, code_body);
	env->ReleaseStringUTFChars(j_code_body, code_body);
	for (int i = 0; i < num_functor_params; i++)
		env->ReleaseStringUTFChars((jstring)env->GetObjectArrayElement(j_functor_params, i), functor_params[i]);
	return jlong(cptr);
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_functor_1create___3J_3Ljava_lang_String_2_3Ljava_lang_String_2Ljava_lang_String_2(JNIEnv * env, jclass, jlongArray p_objs, jobjectArray j_name_objs, jobjectArray j_functor_params, jstring j_code_body)
{
	jsize num_objs = env->GetArrayLength(j_name_objs);
	jlong* lpobjs = env->GetLongArrayElements(p_objs, nullptr);
	std::vector<CapturedDeviceViewable> elem_map(num_objs);
	for (int i = 0; i < num_objs; i++)
	{
		elem_map[i].obj_name = env->GetStringUTFChars((jstring)env->GetObjectArrayElement(j_name_objs, i), nullptr);
		elem_map[i].obj = (const DeviceViewable*)lpobjs[i];
	}
	jsize num_functor_params = env->GetArrayLength(j_functor_params);
	std::vector<const char*> functor_params(num_functor_params);
	for (int i = 0; i < num_functor_params; i++)
		functor_params[i] = env->GetStringUTFChars((jstring)env->GetObjectArrayElement(j_functor_params, i), nullptr);
	const char* code_body = env->GetStringUTFChars(j_code_body, nullptr);
	Functor* cptr = new Functor(elem_map, functor_params, code_body);
	env->ReleaseStringUTFChars(j_code_body, code_body);
	for (int i = 0; i < num_functor_params; i++)
		env->ReleaseStringUTFChars((jstring)env->GetObjectArrayElement(j_functor_params, i), functor_params[i]);
	env->ReleaseLongArrayElements(p_objs, lpobjs, 0);
	for (int i = 0; i < num_objs; i++)
		env->ReleaseStringUTFChars((jstring)env->GetObjectArrayElement(j_name_objs, i), elem_map[i].obj_name);
	return jlong(cptr);
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_built_1in_1functor_1create(JNIEnv *env, jclass, jstring j_name_built_in_view_cls)
{
	const char* name_built_in_view_cls = env->GetStringUTFChars(j_name_built_in_view_cls, nullptr);
	return (jlong)(new Functor(name_built_in_view_cls));
}
