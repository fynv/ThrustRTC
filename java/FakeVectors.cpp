#include "JThrustRTC_Native.h"
#include "fake_vectors/DVConstant.h"
#include "fake_vectors/DVCounter.h"
#include "fake_vectors/DVDiscard.h"
#include "fake_vectors/DVPermutation.h"
#include "fake_vectors/DVReverse.h"
#include "fake_vectors/DVTransform.h"
#include "fake_vectors/DVZipped.h"
#include "fake_vectors/DVCustomVector.h"

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvconstant_1create(JNIEnv *, jclass, jlong p_dvobj, jint size)
{
	DeviceViewable* dvobj = (DeviceViewable*)p_dvobj;
	return (jlong)(new DVConstant(*dvobj, (size_t)size));
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvcounter_1create(JNIEnv *, jclass, jlong p_dvobj_init, jint size)
{
	DeviceViewable* dvobj_init = (DeviceViewable*)p_dvobj_init;
	return (jlong)(new DVCounter(*dvobj_init, (size_t)size));
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvdiscard_1create(JNIEnv *env, jclass, jstring j_elem_cls, jint size)
{
	const char *elem_cls = env->GetStringUTFChars(j_elem_cls, nullptr);
	DVDiscard* cptr = new DVDiscard(elem_cls, (size_t)size);
	env->ReleaseStringUTFChars(j_elem_cls, elem_cls);
	return (jlong)cptr;
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvpermutation_1create(JNIEnv *, jclass, jlong p_vec_value, jlong p_vec_index)
{
	DVVectorLike* vec_value = (DVVectorLike*)p_vec_value;
	DVVectorLike* vec_index = (DVVectorLike*)p_vec_index;
	return (jlong)(new DVPermutation(*vec_value, *vec_index));
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvreverse_1create(JNIEnv *, jclass, jlong p_vec_value)
{
	DVVectorLike* vec_value = (DVVectorLike*)p_vec_value;
	return (jlong)(new DVReverse(*vec_value));
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvtransform_1create(JNIEnv *env, jclass, jlong p_vec_in, jstring j_elem_cls, jlong p_op)
{
	DVVectorLike* vec_in = (DVVectorLike*)p_vec_in;
	Functor* op = (Functor*)p_op;
	const char* elem_cls = env->GetStringUTFChars(j_elem_cls, nullptr);
	DVTransform* cptr = new DVTransform(*vec_in, elem_cls, *op);
	env->ReleaseStringUTFChars(j_elem_cls, elem_cls);
	return (jlong)(cptr);
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvzipped_1create(JNIEnv *env, jclass, jlongArray j_vecs, jobjectArray j_elem_names)
{
	jsize num_vecs = env->GetArrayLength(j_vecs);
	jlong* lpvec = env->GetLongArrayElements(j_vecs, nullptr);
	std::vector<DVVectorLike*> vecs(num_vecs);
	for (int i = 0; i < num_vecs; i++)
		vecs[i] = (DVVectorLike*)lpvec[i];

	jsize num_elems = env->GetArrayLength(j_elem_names);
	if (num_elems != num_vecs) return 0;
	std::vector<const char*> elem_names(num_elems);
	for (int i = 0; i < num_elems; i++)
		elem_names[i] = env->GetStringUTFChars((jstring)env->GetObjectArrayElement(j_elem_names, i), nullptr);

	DVZipped* cptr = new DVZipped(vecs, elem_names);

	env->ReleaseLongArrayElements(j_vecs, lpvec, 0);
	for (int i = 0; i < num_elems; i++)
		env->ReleaseStringUTFChars((jstring)env->GetObjectArrayElement(j_elem_names, i), elem_names[i]);

	return (jlong)cptr;
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvcustomvector_1create
(JNIEnv *env, jclass, jlongArray p_objs, jobjectArray j_name_objs, jstring j_name_idx, jstring j_code_body, jstring j_elem_cls, jint size, jboolean read_only)
{
	jsize num_objs = env->GetArrayLength(j_name_objs);
	jlong* lpobjs = env->GetLongArrayElements(p_objs, nullptr);

	std::vector<CapturedDeviceViewable> elem_map(num_objs);
	for (int i = 0; i < num_objs; i++)
	{
		elem_map[i].obj_name = env->GetStringUTFChars((jstring)env->GetObjectArrayElement(j_name_objs, i), nullptr);
		elem_map[i].obj = (const DeviceViewable*)lpobjs[i];
	}
	const char* name_idx = env->GetStringUTFChars(j_name_idx, nullptr);
	const char* code_body = env->GetStringUTFChars(j_code_body, nullptr);
	const char* elem_cls = env->GetStringUTFChars(j_elem_cls, nullptr);

	DVCustomVector* cptr = new DVCustomVector(elem_map, name_idx, code_body, elem_cls, (size_t)size, read_only != 0);

	env->ReleaseStringUTFChars(j_elem_cls, elem_cls);
	env->ReleaseStringUTFChars(j_code_body, code_body);
	env->ReleaseStringUTFChars(j_name_idx, name_idx);

	env->ReleaseLongArrayElements(p_objs, lpobjs, 0);
	for (int i = 0; i < num_objs; i++)
		env->ReleaseStringUTFChars((jstring)env->GetObjectArrayElement(j_name_objs, i), elem_map[i].obj_name);

	return (jlong)cptr;
}