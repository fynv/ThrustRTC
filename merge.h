#ifndef _TRTC_merge_h
#define _TRTC_merge_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Merge(const DVVectorLike& vec1, const DVVectorLike& vec2, DVVectorLike& vec_out, size_t begin1 = 0, size_t end1 = (size_t)(-1), size_t begin2 = 0, size_t end2 = (size_t)(-1), size_t begin_out = 0);
bool THRUST_RTC_API TRTC_Merge(const DVVectorLike& vec1, const DVVectorLike& vec2, DVVectorLike& vec_out, const Functor& comp, size_t begin1 = 0, size_t end1 = (size_t)(-1), size_t begin2 = 0, size_t end2 = (size_t)(-1), size_t begin_out = 0);
bool THRUST_RTC_API TRTC_Merge_By_Key(const DVVectorLike& keys1, const DVVectorLike& keys2, const DVVectorLike& value1, const DVVectorLike& value2, DVVectorLike& keys_out, DVVectorLike& value_out, size_t begin_keys1 = 0, size_t end_keys1 = (size_t)(-1), size_t begin_keys2 = 0, size_t end_keys2 = (size_t)(-1), size_t begin_value1 = 0, size_t begin_value2 = 0, size_t begin_keys_out = 0, size_t begin_value_out = 0);
bool THRUST_RTC_API TRTC_Merge_By_Key(const DVVectorLike& keys1, const DVVectorLike& keys2, const DVVectorLike& value1, const DVVectorLike& value2, DVVectorLike& keys_out, DVVectorLike& value_out, const Functor& comp, size_t begin_keys1 = 0, size_t end_keys1 = (size_t)(-1), size_t begin_keys2 = 0, size_t end_keys2 = (size_t)(-1), size_t begin_value1 = 0, size_t begin_value2 = 0, size_t begin_keys_out = 0, size_t begin_value_out = 0);

#endif
