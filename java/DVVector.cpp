#include "JThrustRTC_Native.h"
#include "DVVector.h"
#include "fake_vectors/DVRange.h"

JNIEXPORT jstring JNICALL Java_JThrustRTC_Native_dvvectorlike_1name_1elem_1cls(JNIEnv *env, jclass, jlong p_dvvec)
{
	DVVectorLike* dvvec = (DVVectorLike*)p_dvvec;
	return env->NewStringUTF(dvvec->name_elem_cls().c_str());
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_dvvectorlike_1size(JNIEnv *, jclass, jlong p_dvvec)
{
	DVVectorLike* dvvec = (DVVectorLike*)p_dvvec;
	return (jint)dvvec->size();
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvrange_1create(JNIEnv *, jclass, jlong p_dvvec, jint begin, jint end)
{
	DVVectorLike* dvvec = (DVVectorLike*)p_dvvec;
	DVVectorLike* vec_value = (DVVectorLike*)p_dvvec;

	DVVector* p_vec = dynamic_cast<DVVector*>(vec_value);
	if (p_vec)
		return (jlong)(new DVVectorAdaptor(*p_vec, (size_t)begin, (size_t)end));

	DVVectorAdaptor* p_vec_adpt = dynamic_cast<DVVectorAdaptor*>(vec_value);
	if (p_vec_adpt)
		return (jlong)(new DVVectorAdaptor(*p_vec_adpt, (size_t)begin, (size_t)end));

	return (jlong)(new DVRange(*vec_value, (size_t)begin, (size_t)end));
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create__Ljava_lang_String_2I(JNIEnv *env, jclass, jstring j_elem_cls, jint size)
{
	const char *elem_cls = env->GetStringUTFChars(j_elem_cls, nullptr);
	DVVector* cptr = new DVVector(elem_cls, (size_t)size);
	env->ReleaseStringUTFChars(j_elem_cls, elem_cls);
	return (jlong)cptr;
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3B(JNIEnv *env, jclass, jbyteArray hdata)
{
	jbyte* p_hdata = env->GetByteArrayElements(hdata, nullptr);
	DVVector* cptr = new DVVector("int8_t", env->GetArrayLength(hdata), p_hdata);
	env->ReleaseByteArrayElements(hdata, p_hdata, 0);
	return (jlong)cptr;
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3S(JNIEnv *env, jclass, jshortArray hdata)
{
	jshort* p_hdata = env->GetShortArrayElements(hdata, nullptr);
	DVVector* cptr = new DVVector("int16_t", env->GetArrayLength(hdata), p_hdata);
	env->ReleaseShortArrayElements(hdata, p_hdata, 0);
	return (jlong)cptr;
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3I(JNIEnv *env, jclass, jintArray hdata)
{
	jint* p_hdata = env->GetIntArrayElements(hdata, nullptr);
	DVVector* cptr = new DVVector("int32_t", env->GetArrayLength(hdata), p_hdata);
	env->ReleaseIntArrayElements(hdata, p_hdata, 0);
	return (jlong)cptr;
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3J(JNIEnv *env, jclass, jlongArray hdata)
{
	jlong* p_hdata = env->GetLongArrayElements(hdata, nullptr);
	DVVector* cptr = new DVVector("int64_t", env->GetArrayLength(hdata), env->GetLongArrayElements(hdata, nullptr));
	env->ReleaseLongArrayElements(hdata, p_hdata, 0);
	return (jlong)cptr;
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3F(JNIEnv *env, jclass, jfloatArray hdata)
{
	jfloat* p_hdata = env->GetFloatArrayElements(hdata, nullptr);
	DVVector* cptr = new DVVector("float", env->GetArrayLength(hdata), env->GetFloatArrayElements(hdata, nullptr));
	env->ReleaseFloatArrayElements(hdata, p_hdata, 0);
	return (jlong)cptr;
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3D(JNIEnv *env, jclass, jdoubleArray hdata)
{
	jdouble* p_hdata = env->GetDoubleArrayElements(hdata, nullptr);
	DVVector* cptr = new DVVector("double", env->GetArrayLength(hdata), env->GetDoubleArrayElements(hdata, nullptr));
	env->ReleaseDoubleArrayElements(hdata, p_hdata, 0);
	return (jlong)cptr;
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3BII(JNIEnv *env, jclass, jlong p_dvvec, jbyteArray hdata, jint begin, jint end)
{
	DVVector* dvvec = (DVVector*)p_dvvec;
	jbyte* p_hdata = env->GetByteArrayElements(hdata, nullptr);
	dvvec->to_host(p_hdata, (size_t)begin, (size_t)end);
	env->ReleaseByteArrayElements(hdata, p_hdata, 0);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3SII(JNIEnv *env, jclass, jlong p_dvvec, jshortArray hdata, jint begin, jint end)
{
	DVVector* dvvec = (DVVector*)p_dvvec;
	jshort* p_hdata = env->GetShortArrayElements(hdata, nullptr);
	dvvec->to_host(p_hdata, (size_t)begin, (size_t)end);
	env->ReleaseShortArrayElements(hdata, p_hdata, 0);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3III(JNIEnv *env, jclass, jlong p_dvvec, jintArray hdata, jint begin, jint end)
{
	DVVector* dvvec = (DVVector*)p_dvvec;
	jint* p_hdata = env->GetIntArrayElements(hdata, nullptr);
	dvvec->to_host(p_hdata, (size_t)begin, (size_t)end);
	env->ReleaseIntArrayElements(hdata, p_hdata, 0);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3JII(JNIEnv *env, jclass, jlong p_dvvec, jlongArray hdata, jint begin, jint end)
{
	DVVector* dvvec = (DVVector*)p_dvvec;
	jlong* p_hdata = env->GetLongArrayElements(hdata, nullptr);
	dvvec->to_host(p_hdata, (size_t)begin, (size_t)end);
	env->ReleaseLongArrayElements(hdata, p_hdata, 0);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3FII(JNIEnv *env, jclass, jlong p_dvvec, jfloatArray hdata, jint begin, jint end)
{
	DVVector* dvvec = (DVVector*)p_dvvec;
	float* p_hdata = env->GetFloatArrayElements(hdata, nullptr);
	dvvec->to_host(p_hdata, (size_t)begin, (size_t)end);
	env->ReleaseFloatArrayElements(hdata, p_hdata, 0);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3DII(JNIEnv *env, jclass, jlong p_dvvec, jdoubleArray hdata, jint begin, jint end)
{
	DVVector* dvvec = (DVVector*)p_dvvec;
	double* p_hdata = env->GetDoubleArrayElements(hdata, nullptr);
	dvvec->to_host(p_hdata, (size_t)begin, (size_t)end);
	env->ReleaseDoubleArrayElements(hdata, p_hdata, 0);
}
