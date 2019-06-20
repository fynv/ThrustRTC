#include "JThrustRTC_Native.h"
#include "DeviceViewable.h"

JNIEXPORT jstring JNICALL Java_JThrustRTC_Native_dv_1name_1view_1cls(JNIEnv *env, jclass, jlong p_dv)
{
	DeviceViewable* dv = (DeviceViewable*)p_dv;
	return env->NewStringUTF(dv->name_view_cls().c_str());
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_dv_1destroy(JNIEnv *, jclass, jlong p_dv)
{
	DeviceViewable* dv = (DeviceViewable*)p_dv;
	delete dv;
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvint8_1create(JNIEnv *, jclass, jbyte v)
{
	return (jlong)(new DVInt8(v));
}

JNIEXPORT jbyte JNICALL Java_JThrustRTC_Native_dvint8_1value(JNIEnv *, jclass, jlong p)
{
	return *(jbyte*)((DVInt8*)p)->view().data();
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvint16_1create(JNIEnv *, jclass, jshort v)
{
	return (jlong)(new DVInt16(v));
}

JNIEXPORT jshort JNICALL Java_JThrustRTC_Native_dvint16_1value(JNIEnv *, jclass, jlong p)
{
	return *(jshort*)((DVInt16*)p)->view().data();
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvint32_1create(JNIEnv *, jclass, jint v)
{
	return (jlong)(new DVInt32(v));
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_dvint32_1value(JNIEnv *, jclass, jlong p)
{
	return *(jint*)((DVInt32*)p)->view().data();
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvint64_1create(JNIEnv *, jclass, jlong v)
{
	return (jlong)(new DVInt64(v));
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvint64_1value(JNIEnv *, jclass, jlong p)
{
	return *(jlong*)((DVInt64*)p)->view().data();
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvfloat_1create(JNIEnv *, jclass, jfloat v)
{
	return (jlong)(new DVFloat(v));
}

JNIEXPORT jfloat JNICALL Java_JThrustRTC_Native_dvfloat_1value(JNIEnv *, jclass, jlong p)
{
	return *(jfloat*)((DVFloat*)p)->view().data();
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvdouble_1create(JNIEnv *, jclass, jdouble v)
{
	return (jlong)(new DVDouble(v));
}

JNIEXPORT jdouble JNICALL Java_JThrustRTC_Native_dvdouble_1value(JNIEnv *, jclass, jlong p)
{
	return *(jdouble*)((DVDouble*)p)->view().data();
}

