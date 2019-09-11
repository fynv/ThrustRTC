#include "JThrustRTC_Native.h"
#include "TRTCContext.h"
#include <vector>
#include <memory.h>

JNIEXPORT void JNICALL Java_JThrustRTC_Native_set_1libnvrtc_1path (JNIEnv * env, jclass, jstring j_path)
{
	const char *path = env->GetStringUTFChars(j_path, nullptr);
	set_libnvrtc_path(path);	
	env->ReleaseStringUTFChars(j_path, path);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_set_1verbose(JNIEnv *, jclass, jboolean j_verbose)
{
	TRTC_Set_Verbose(j_verbose!=0);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_add_1include_1dir(JNIEnv * env, jclass, jstring j_dir)
{
	const char *dir = env->GetStringUTFChars(j_dir, nullptr);
	TRTC_Add_Include_Dir(dir);
	env->ReleaseStringUTFChars(j_dir, dir);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_add_1built_1in_1header(JNIEnv * env, jclass, jstring j_filename, jstring j_filecontent)
{
	const char *filename = env->GetStringUTFChars(j_filename, nullptr);
	const char *filecontent = env->GetStringUTFChars(j_filecontent, nullptr);
	TRTC_Add_Built_In_Header(filename, filecontent);
	//deliberately not released
	//env->ReleaseStringUTFChars(j_filename, filename);
	//env->ReleaseStringUTFChars(j_filecontent, filecontent);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_add_1include_1filename(JNIEnv * env, jclass, jstring j_filename)
{
	const char *filename = env->GetStringUTFChars(j_filename, nullptr);
	TRTC_Add_Inlcude_Filename(filename);
	env->ReleaseStringUTFChars(j_filename, filename);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_add_1code_1block(JNIEnv *env, jclass, jstring j_code)
{
	const char *code = env->GetStringUTFChars(j_code, nullptr);
	TRTC_Add_Code_Block(code);
	env->ReleaseStringUTFChars(j_code, code);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_sync(JNIEnv *, jclass)
{
	TRTC_Wait();
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_kernel_1create(JNIEnv *env, jclass, jobjectArray param_names, jstring j_body)
{
	jsize num_params = env->GetArrayLength(param_names);
	std::vector<const char*> params(num_params);
	for (int i = 0; i < num_params; i++)
		params[i] = env->GetStringUTFChars((jstring)env->GetObjectArrayElement(param_names, i), nullptr);
	const char* body = env->GetStringUTFChars(j_body, nullptr);
	TRTC_Kernel* cptr = new TRTC_Kernel(params, body);
	env->ReleaseStringUTFChars(j_body, body);
	for (int i = 0; i < num_params; i++)
		env->ReleaseStringUTFChars((jstring)env->GetObjectArrayElement(param_names, i), params[i]);
	return jlong(cptr);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_kernel_1destroy(JNIEnv *, jclass, jlong p_kernel)
{
	TRTC_Kernel* kernel = (TRTC_Kernel*)p_kernel;
	delete kernel;
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_kernel_1num_1params(JNIEnv *, jclass, jlong p_kernel)
{
	TRTC_Kernel* kernel = (TRTC_Kernel*)p_kernel;
	return (jint)kernel->num_params();
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_kernel_1calc_1optimal_1block_1size(JNIEnv *env, jclass, jlong p_kernel, jlongArray p_args, jint sharedMemBytes)
{
	TRTC_Kernel* kernel = (TRTC_Kernel*)p_kernel;
	jlong* lpargs= env->GetLongArrayElements(p_args, nullptr);
	const DeviceViewable** args = (const DeviceViewable**)lpargs;
	int sizeBlock;
	if (!kernel->calc_optimal_block_size(args, sizeBlock, (unsigned)sharedMemBytes)) sizeBlock = -1;
	env->ReleaseLongArrayElements(p_args, lpargs, 0);
	return sizeBlock;
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_kernel_1calc_1number_1blocks(JNIEnv *env, jclass, jlong p_kernel, jlongArray p_args, jint sizeBlock, jint sharedMemBytes)
{
	TRTC_Kernel* kernel = (TRTC_Kernel*)p_kernel;
	jlong* lpargs = env->GetLongArrayElements(p_args, nullptr);
	const DeviceViewable** args = (const DeviceViewable**)lpargs;
	int numBlocks;
	if (!kernel->calc_number_blocks(args, sizeBlock, numBlocks, (unsigned)sharedMemBytes)) numBlocks = -1;
	env->ReleaseLongArrayElements(p_args, lpargs, 0);
	return numBlocks;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_kernel_1launch(JNIEnv *env, jclass, jlong p_kernel, jintArray gridDim, jintArray blockDim, jlongArray p_args, jint sharedMemBytes)
{
	TRTC_Kernel* kernel = (TRTC_Kernel*)p_kernel;
	jlong* lpargs = env->GetLongArrayElements(p_args, nullptr);
	const DeviceViewable** args = (const DeviceViewable**)lpargs;
	jsize size_gridDim = env->GetArrayLength(gridDim);
	jsize size_blockDim = env->GetArrayLength(blockDim);
	jint* p_gridDim = env->GetIntArrayElements(gridDim, nullptr);
	jint* p_blockDim = env->GetIntArrayElements(blockDim, nullptr);

	dim_type c_gridDim = { 1, 1, 1 };
	if (size_gridDim > 0) c_gridDim.x = p_gridDim[0];
	if (size_gridDim > 1) c_gridDim.y = p_gridDim[1];
	if (size_gridDim > 2) c_gridDim.z = p_gridDim[2];

	dim_type c_blockDim = { 1, 1, 1 };
	if (size_blockDim > 0) c_blockDim.x = p_blockDim[0];
	if (size_blockDim > 1) c_blockDim.y = p_blockDim[1];
	if (size_blockDim > 2) c_blockDim.z = p_blockDim[2];

	jboolean res = kernel->launch(c_gridDim, c_blockDim, args, sharedMemBytes) ? 1 : 0;
	env->ReleaseIntArrayElements(blockDim, p_blockDim, 0);
	env->ReleaseIntArrayElements(gridDim, p_gridDim, 0);
	env->ReleaseLongArrayElements(p_args, lpargs, 0);

	return res;
}

JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_for_1create(JNIEnv *env, jclass, jobjectArray param_names, jstring j_name_iter, jstring j_body)
{
	jsize num_params = env->GetArrayLength(param_names);
	std::vector<const char*> params(num_params);
	for (int i = 0; i < num_params; i++)
		params[i] = env->GetStringUTFChars((jstring)env->GetObjectArrayElement(param_names, i), nullptr);
	const char* name_iter = env->GetStringUTFChars(j_name_iter, nullptr);
	const char* body = env->GetStringUTFChars(j_body, nullptr);
	TRTC_For* cptr = new TRTC_For(params, name_iter, body);
	env->ReleaseStringUTFChars(j_body, body);
	env->ReleaseStringUTFChars(j_name_iter, name_iter);
	for (int i = 0; i < num_params; i++)
		env->ReleaseStringUTFChars((jstring)env->GetObjectArrayElement(param_names, i), params[i]);
	return jlong(cptr);
}

JNIEXPORT void JNICALL Java_JThrustRTC_Native_for_1destroy(JNIEnv *env, jclass, jlong p_kernel)
{
	TRTC_For* kernel = (TRTC_For*)p_kernel;
	delete kernel;
}

JNIEXPORT jint JNICALL Java_JThrustRTC_Native_for_1num_1params(JNIEnv *, jclass, jlong p_kernel)
{
	TRTC_For* kernel = (TRTC_For*)p_kernel;
	return (jint)kernel->num_params();
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_for_1launch(JNIEnv *env, jclass, jlong p_kernel, jint begin, jint end, jlongArray p_args)
{
	TRTC_For* kernel = (TRTC_For*)p_kernel;
	jlong* lpargs = env->GetLongArrayElements(p_args, nullptr);
	const DeviceViewable** args = (const DeviceViewable**)lpargs;
	jboolean res = kernel->launch((size_t)begin, (size_t)end, args) ? 1 : 0;
	env->ReleaseLongArrayElements(p_args, lpargs, 0);
	return res;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_for_1launch_1n(JNIEnv *env, jclass, jlong p_kernel, jint n, jlongArray p_args)
{
	TRTC_For* kernel = (TRTC_For*)p_kernel;
	jlong* lpargs = env->GetLongArrayElements(p_args, nullptr);
	const DeviceViewable** args = (const DeviceViewable**)lpargs;
	jboolean res = kernel->launch_n((size_t)n, args) ? 1 : 0;
	env->ReleaseLongArrayElements(p_args, lpargs, 0);
	return res;
}
