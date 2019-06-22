#include "JThrustRTC_Native.h"
#include "gather.h"
#include "scatter.h"
#include "copy.h"
#include "swap.h"

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_gather(JNIEnv *, jclass, jlong p_vec_map, jlong p_vec_in, jlong p_vec_out)
{
	DVVectorLike* vec_map = (DVVectorLike*)(p_vec_map);
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return TRTC_Gather(*vec_map, *vec_in, *vec_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_gather_1if__JJJJ(JNIEnv *, jclass, jlong p_vec_map, jlong p_vec_stencil, jlong p_vec_in, jlong p_vec_out)
{
	DVVectorLike* vec_map = (DVVectorLike*)(p_vec_map);
	DVVectorLike* vec_stencil = (DVVectorLike*)(p_vec_stencil);
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return TRTC_Gather_If(*vec_map, *vec_stencil, *vec_in, *vec_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_gather_1if__JJJJJ(JNIEnv *, jclass, jlong p_vec_map, jlong p_vec_stencil, jlong p_vec_in, jlong p_vec_out, jlong p_pred)
{
	DVVectorLike* vec_map = (DVVectorLike*)(p_vec_map);
	DVVectorLike* vec_stencil = (DVVectorLike*)(p_vec_stencil);
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* pred = (Functor*)(p_pred);
	return TRTC_Gather_If(*vec_map, *vec_stencil, *vec_in, *vec_out, *pred) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_scatter(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_map, jlong p_vec_out)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_map = (DVVectorLike*)(p_vec_map);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return TRTC_Scatter(*vec_in, *vec_map, *vec_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_scatter_1if__JJJJ(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_map, jlong p_vec_stencil, jlong p_vec_out)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_map = (DVVectorLike*)(p_vec_map);
	DVVectorLike* vec_stencil = (DVVectorLike*)(p_vec_stencil);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return TRTC_Scatter_If(*vec_in, *vec_map, *vec_stencil, *vec_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_scatter_1if__JJJJJ(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_map, jlong p_vec_stencil, jlong p_vec_out, jlong p_pred)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_map = (DVVectorLike*)(p_vec_map);
	DVVectorLike* vec_stencil = (DVVectorLike*)(p_vec_stencil);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	Functor* pred = (Functor*)(p_pred);
	return TRTC_Scatter_If(*vec_in, *vec_map, *vec_stencil, *vec_out, *pred) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_copy(JNIEnv *, jclass, jlong p_vec_in, jlong p_vec_out)
{
	DVVectorLike* vec_in = (DVVectorLike*)(p_vec_in);
	DVVectorLike* vec_out = (DVVectorLike*)(p_vec_out);
	return TRTC_Copy(*vec_in, *vec_out) ? 1 : 0;
}

JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_swap(JNIEnv *, jclass, jlong p_vec1, jlong p_vec2)
{
	DVVectorLike* vec1 = (DVVectorLike*)(p_vec1);
	DVVectorLike* vec2 = (DVVectorLike*)(p_vec2);
	return TRTC_Swap(*vec1, *vec2) ? 1 : 0;
}
