/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class JThrustRTC_Native */

#ifndef _Included_JThrustRTC_Native
#define _Included_JThrustRTC_Native
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     JThrustRTC_Native
 * Method:    set_libnvrtc_path
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_set_1libnvrtc_1path
  (JNIEnv *, jclass, jstring);

/*
 * Class:     JThrustRTC_Native
 * Method:    set_verbose
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_set_1verbose
  (JNIEnv *, jclass, jboolean);

/*
 * Class:     JThrustRTC_Native
 * Method:    add_include_dir
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_add_1include_1dir
  (JNIEnv *, jclass, jstring);

/*
 * Class:     JThrustRTC_Native
 * Method:    add_built_in_header
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_add_1built_1in_1header
  (JNIEnv *, jclass, jstring, jstring);

/*
 * Class:     JThrustRTC_Native
 * Method:    add_include_filename
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_add_1include_1filename
  (JNIEnv *, jclass, jstring);

/*
 * Class:     JThrustRTC_Native
 * Method:    add_code_block
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_add_1code_1block
  (JNIEnv *, jclass, jstring);

/*
 * Class:     JThrustRTC_Native
 * Method:    kernel_create
 * Signature: ([Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_kernel_1create
  (JNIEnv *, jclass, jobjectArray, jstring);

/*
 * Class:     JThrustRTC_Native
 * Method:    kernel_destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_kernel_1destroy
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    kernel_num_params
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_kernel_1num_1params
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    kernel_calc_optimal_block_size
 * Signature: (J[JI)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_kernel_1calc_1optimal_1block_1size
  (JNIEnv *, jclass, jlong, jlongArray, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    kernel_calc_number_blocks
 * Signature: (J[JII)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_kernel_1calc_1number_1blocks
  (JNIEnv *, jclass, jlong, jlongArray, jint, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    kernel_launch
 * Signature: (J[I[I[JI)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_kernel_1launch
  (JNIEnv *, jclass, jlong, jintArray, jintArray, jlongArray, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    for_create
 * Signature: ([Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_for_1create
  (JNIEnv *, jclass, jobjectArray, jstring, jstring);

/*
 * Class:     JThrustRTC_Native
 * Method:    for_destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_for_1destroy
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    for_num_params
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_for_1num_1params
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    for_launch
 * Signature: (JII[J)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_for_1launch
  (JNIEnv *, jclass, jlong, jint, jint, jlongArray);

/*
 * Class:     JThrustRTC_Native
 * Method:    for_launch_n
 * Signature: (JI[J)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_for_1launch_1n
  (JNIEnv *, jclass, jlong, jint, jlongArray);

/*
 * Class:     JThrustRTC_Native
 * Method:    dv_name_view_cls
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_JThrustRTC_Native_dv_1name_1view_1cls
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dv_destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_dv_1destroy
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvint8_create
 * Signature: (B)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvint8_1create
  (JNIEnv *, jclass, jbyte);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvint8_value
 * Signature: (J)B
 */
JNIEXPORT jbyte JNICALL Java_JThrustRTC_Native_dvint8_1value
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvint16_create
 * Signature: (S)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvint16_1create
  (JNIEnv *, jclass, jshort);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvint16_value
 * Signature: (J)S
 */
JNIEXPORT jshort JNICALL Java_JThrustRTC_Native_dvint16_1value
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvint32_create
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvint32_1create
  (JNIEnv *, jclass, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvint32_value
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_dvint32_1value
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvint64_create
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvint64_1create
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvint64_value
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvint64_1value
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvfloat_create
 * Signature: (F)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvfloat_1create
  (JNIEnv *, jclass, jfloat);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvfloat_value
 * Signature: (J)F
 */
JNIEXPORT jfloat JNICALL Java_JThrustRTC_Native_dvfloat_1value
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvdouble_create
 * Signature: (D)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvdouble_1create
  (JNIEnv *, jclass, jdouble);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvdouble_value
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_JThrustRTC_Native_dvdouble_1value
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvectorlike_name_elem_cls
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_JThrustRTC_Native_dvvectorlike_1name_1elem_1cls
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvectorlike_size
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_dvvectorlike_1size
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvrange_create
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvrange_1create
  (JNIEnv *, jclass, jlong, jint, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_create
 * Signature: (Ljava/lang/String;I)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create__Ljava_lang_String_2I
  (JNIEnv *, jclass, jstring, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_create
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3B
  (JNIEnv *, jclass, jbyteArray);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_create
 * Signature: ([S)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3S
  (JNIEnv *, jclass, jshortArray);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_create
 * Signature: ([I)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3I
  (JNIEnv *, jclass, jintArray);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_create
 * Signature: ([J)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3J
  (JNIEnv *, jclass, jlongArray);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_create
 * Signature: ([F)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3F
  (JNIEnv *, jclass, jfloatArray);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_create
 * Signature: ([D)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvvector_1create___3D
  (JNIEnv *, jclass, jdoubleArray);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_to_host
 * Signature: (J[BII)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3BII
  (JNIEnv *, jclass, jlong, jbyteArray, jint, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_to_host
 * Signature: (J[SII)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3SII
  (JNIEnv *, jclass, jlong, jshortArray, jint, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_to_host
 * Signature: (J[III)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3III
  (JNIEnv *, jclass, jlong, jintArray, jint, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_to_host
 * Signature: (J[JII)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3JII
  (JNIEnv *, jclass, jlong, jlongArray, jint, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_to_host
 * Signature: (J[FII)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3FII
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvvector_to_host
 * Signature: (J[DII)V
 */
JNIEXPORT void JNICALL Java_JThrustRTC_Native_dvvector_1to_1host__J_3DII
  (JNIEnv *, jclass, jlong, jdoubleArray, jint, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvtuple_create
 * Signature: ([J[Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvtuple_1create
  (JNIEnv *, jclass, jlongArray, jobjectArray);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvconstant_create
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvconstant_1create
  (JNIEnv *, jclass, jlong, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvcounter_create
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvcounter_1create
  (JNIEnv *, jclass, jlong, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvdiscard_create
 * Signature: (Ljava/lang/String;I)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvdiscard_1create
  (JNIEnv *, jclass, jstring, jint);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvpermutation_create
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvpermutation_1create
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvreverse_create
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvreverse_1create
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvtransform_create
 * Signature: (JLjava/lang/String;J)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvtransform_1create
  (JNIEnv *, jclass, jlong, jstring, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvzipped_create
 * Signature: ([J[Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvzipped_1create
  (JNIEnv *, jclass, jlongArray, jobjectArray);

/*
 * Class:     JThrustRTC_Native
 * Method:    dvcustomvector_create
 * Signature: ([J[Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;IZ)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_dvcustomvector_1create
  (JNIEnv *, jclass, jlongArray, jobjectArray, jstring, jstring, jstring, jint, jboolean);

/*
 * Class:     JThrustRTC_Native
 * Method:    functor_create
 * Signature: ([Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_functor_1create___3Ljava_lang_String_2Ljava_lang_String_2
  (JNIEnv *, jclass, jobjectArray, jstring);

/*
 * Class:     JThrustRTC_Native
 * Method:    functor_create
 * Signature: ([J[Ljava/lang/String;[Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_functor_1create___3J_3Ljava_lang_String_2_3Ljava_lang_String_2Ljava_lang_String_2
  (JNIEnv *, jclass, jlongArray, jobjectArray, jobjectArray, jstring);

/*
 * Class:     JThrustRTC_Native
 * Method:    built_in_functor_create
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_JThrustRTC_Native_built_1in_1functor_1create
  (JNIEnv *, jclass, jstring);

/*
 * Class:     JThrustRTC_Native
 * Method:    fill
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_fill
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    replace
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_replace
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    replace_if
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_replace_1if
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    replace_copy
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_replace_1copy
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    replace_copy_if
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_replace_1copy_1if
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    for_each
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_for_1each
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    adjacent_difference
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_adjacent_1difference__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    adjacent_difference
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_adjacent_1difference__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    sequence
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sequence__J
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    sequence
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sequence__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    sequence
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sequence__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    tabulate
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_tabulate
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    transform
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    transform_binary
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1binary
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    transform_if
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1if
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    transform_if_stencil
 * Signature: (JJJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1if_1stencil
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    transform_binary_if_stencil
 * Signature: (JJJJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1binary_1if_1stencil
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    gather
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_gather
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    gather_if
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_gather_1if__JJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    gather_if
 * Signature: (JJJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_gather_1if__JJJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    scatter
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_scatter
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    scatter_if
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_scatter_1if__JJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    scatter_if
 * Signature: (JJJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_scatter_1if__JJJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    copy
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_copy
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    swap
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_swap
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    count
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_count
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    count_if
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_count_1if
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    reduce
 * Signature: (J)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_reduce__J
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    reduce
 * Signature: (JJ)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_reduce__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    reduce
 * Signature: (JJJ)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_reduce__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    reduce_by_key
 * Signature: (JJJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_reduce_1by_1key__JJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    reduce_by_key
 * Signature: (JJJJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_reduce_1by_1key__JJJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    reduce_by_key
 * Signature: (JJJJJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_reduce_1by_1key__JJJJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    equal
 * Signature: (JJ)Ljava/lang/Boolean;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_equal__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    equal
 * Signature: (JJJ)Ljava/lang/Boolean;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_equal__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    min_element
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_min_1element__J
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    min_element
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_min_1element__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    max_element
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_max_1element__J
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    max_element
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_max_1element__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    minmax_element
 * Signature: (J)[I
 */
JNIEXPORT jintArray JNICALL Java_JThrustRTC_Native_minmax_1element__J
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    minmax_element
 * Signature: (JJ)[I
 */
JNIEXPORT jintArray JNICALL Java_JThrustRTC_Native_minmax_1element__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    inner_product
 * Signature: (JJJ)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_inner_1product__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    inner_product
 * Signature: (JJJJJ)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_inner_1product__JJJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    transform_reduce
 * Signature: (JJJJ)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_transform_1reduce
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    all_of
 * Signature: (JJ)Ljava/lang/Boolean;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_all_1of
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    any_of
 * Signature: (JJ)Ljava/lang/Boolean;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_any_1of
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    none_of
 * Signature: (JJ)Ljava/lang/Boolean;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_none_1of
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    is_partitioned
 * Signature: (JJ)Ljava/lang/Boolean;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_is_1partitioned
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    is_sorted
 * Signature: (J)Ljava/lang/Boolean;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_is_1sorted__J
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    is_sorted
 * Signature: (JJ)Ljava/lang/Boolean;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_is_1sorted__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    inclusive_scan
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_inclusive_1scan__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    inclusive_scan
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_inclusive_1scan__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    exclusive_scan
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    exclusive_scan
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    exclusive_scan
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan__JJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    inclusive_scan_by_key
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_inclusive_1scan_1by_1key__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    inclusive_scan_by_key
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_inclusive_1scan_1by_1key__JJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    inclusive_scan_by_key
 * Signature: (JJJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_inclusive_1scan_1by_1key__JJJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    exclusive_scan_by_key
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan_1by_1key__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    exclusive_scan_by_key
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan_1by_1key__JJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    exclusive_scan_by_key
 * Signature: (JJJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan_1by_1key__JJJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    exclusive_scan_by_key
 * Signature: (JJJJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_exclusive_1scan_1by_1key__JJJJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    transform_inclusive_scan
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1inclusive_1scan
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    transform_exclusive_scan
 * Signature: (JJJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_transform_1exclusive_1scan
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    copy_if
 * Signature: (JJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_copy_1if
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    copy_if_stencil
 * Signature: (JJJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_copy_1if_1stencil
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    remove
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    remove_copy
 * Signature: (JJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove_1copy
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    remove_if
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove_1if
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    remove_copy_if
 * Signature: (JJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove_1copy_1if
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    remove_if_stencil
 * Signature: (JJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove_1if_1stencil
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    remove_copy_if_stencil
 * Signature: (JJJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_remove_1copy_1if_1stencil
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    unique
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique__J
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    unique
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    unique_copy
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1copy__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    unique_copy
 * Signature: (JJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1copy__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    unique_by_key
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1by_1key__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    unique_by_key
 * Signature: (JJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1by_1key__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    unique_by_key_copy
 * Signature: (JJJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1by_1key_1copy__JJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    unique_by_key_copy
 * Signature: (JJJJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_unique_1by_1key_1copy__JJJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    partition
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_partition
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    partition_stencil
 * Signature: (JJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_partition_1stencil
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    partition_copy
 * Signature: (JJJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_partition_1copy
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    partition_copy_stencil
 * Signature: (JJJJJ)I
 */
JNIEXPORT jint JNICALL Java_JThrustRTC_Native_partition_1copy_1stencil
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    find
 * Signature: (JJ)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_find
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    find_if
 * Signature: (JJ)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_find_1if
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    find_if_not
 * Signature: (JJ)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_find_1if_1not
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    mismatch
 * Signature: (JJ)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_mismatch__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    mismatch
 * Signature: (JJJ)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_mismatch__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    lower_bound
 * Signature: (JJ)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_lower_1bound__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    lower_bound
 * Signature: (JJJ)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_lower_1bound__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    upper_bound
 * Signature: (JJ)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_upper_1bound__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    upper_bound
 * Signature: (JJJ)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_upper_1bound__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    binary_search
 * Signature: (JJ)Ljava/lang/Boolean;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_binary_1search__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    binary_search
 * Signature: (JJJ)Ljava/lang/Boolean;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_binary_1search__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    lower_bound_v
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_lower_1bound_1v__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    lower_bound_v
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_lower_1bound_1v__JJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    upper_bound_v
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_upper_1bound_1v__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    upper_bound_v
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_upper_1bound_1v__JJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    binary_search_v
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_binary_1search_1v__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    binary_search_v
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_binary_1search_1v__JJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    partition_point
 * Signature: (JJ)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_partition_1point
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    is_sorted_until
 * Signature: (J)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_is_1sorted_1until__J
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    is_sorted_until
 * Signature: (JJ)Ljava/lang/Integer;
 */
JNIEXPORT jobject JNICALL Java_JThrustRTC_Native_is_1sorted_1until__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    merge
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_merge__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    merge
 * Signature: (JJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_merge__JJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    merge_by_key
 * Signature: (JJJJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_merge_1by_1key__JJJJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    merge_by_key
 * Signature: (JJJJJJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_merge_1by_1key__JJJJJJJ
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    sort
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sort__J
  (JNIEnv *, jclass, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    sort
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sort__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    sort_by_key
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sort_1by_1key__JJ
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     JThrustRTC_Native
 * Method:    sort_by_key
 * Signature: (JJJ)Z
 */
JNIEXPORT jboolean JNICALL Java_JThrustRTC_Native_sort_1by_1key__JJJ
  (JNIEnv *, jclass, jlong, jlong, jlong);

#ifdef __cplusplus
}
#endif
#endif
