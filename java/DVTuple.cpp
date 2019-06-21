#include "JThrustRTC_Native.h"
#include "DVTuple.h"

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvtuple_1create(JNIEnv *env, jclass, jlongArray p_objs, jobjectArray j_name_objs)
{
	jsize num_objs = env->GetArrayLength(j_name_objs);
	jlong* lpobjs = env->GetLongArrayElements(p_objs, nullptr);

	std::vector<CapturedDeviceViewable> elem_map(num_objs);
	for (int i = 0; i < num_objs; i++)
	{
		elem_map[i].obj_name = env->GetStringUTFChars((jstring)env->GetObjectArrayElement(j_name_objs, i), nullptr);
		elem_map[i].obj = (const DeviceViewable*)lpobjs[i];
	}

	DVTuple* cptr = new DVTuple(elem_map);

	env->ReleaseLongArrayElements(p_objs, lpobjs, 0);
	for (int i = 0; i < num_objs; i++)
		env->ReleaseStringUTFChars((jstring)env->GetObjectArrayElement(j_name_objs, i), elem_map[i].obj_name);

	return (jlong)cptr;
}

