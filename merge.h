#ifndef _TRTC_merge_h
#define _TRTC_merge_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Merge(const DVVectorLike& vec1, const DVVectorLike& vec2, DVVectorLike& vec_out);
bool THRUST_RTC_API TRTC_Merge(const DVVectorLike& vec1, const DVVectorLike& vec2, DVVectorLike& vec_out, const Functor& comp);
bool THRUST_RTC_API TRTC_Merge_By_Key(const DVVectorLike& keys1, const DVVectorLike& keys2, const DVVectorLike& value1, const DVVectorLike& value2, DVVectorLike& keys_out, DVVectorLike& value_out);
bool THRUST_RTC_API TRTC_Merge_By_Key(const DVVectorLike& keys1, const DVVectorLike& keys2, const DVVectorLike& value1, const DVVectorLike& value2, DVVectorLike& keys_out, DVVectorLike& value_out, const Functor& comp);

#endif
