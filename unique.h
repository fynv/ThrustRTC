#ifndef _TRTC_unique_h
#define _TRTC_unique_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

uint32_t THRUST_RTC_API TRTC_Unique(TRTCContext& ctx, DVVectorLike& vec, size_t begin = 0, size_t end = (size_t)(-1));
uint32_t THRUST_RTC_API TRTC_Unique(TRTCContext& ctx, DVVectorLike& vec, const Functor& binary_pred, size_t begin = 0, size_t end = (size_t)(-1));
uint32_t THRUST_RTC_API TRTC_Unique_Copy(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);
uint32_t THRUST_RTC_API TRTC_Unique_Copy(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);
uint32_t THRUST_RTC_API TRTC_Unique_By_Key(TRTCContext& ctx, DVVectorLike& keys, DVVectorLike& values, size_t begin_key = 0, size_t end_key = (size_t)(-1), size_t begin_value = 0);
uint32_t THRUST_RTC_API TRTC_Unique_By_Key(TRTCContext& ctx, DVVectorLike& keys, DVVectorLike& values, const Functor& binary_pred, size_t begin_key = 0, size_t end_key = (size_t)(-1), size_t begin_value = 0);
uint32_t THRUST_RTC_API TRTC_Unique_By_Key_Copy(TRTCContext& ctx, const DVVectorLike& keys_in, const DVVectorLike& values_in, DVVectorLike& keys_out, DVVectorLike& values_out, size_t begin_key_in = 0, size_t end_key_in = (size_t)(-1), size_t begin_value_in = 0, size_t begin_key_out = 0, size_t begin_value_out = 0);
uint32_t THRUST_RTC_API TRTC_Unique_By_Key_Copy(TRTCContext& ctx, const DVVectorLike& keys_in, const DVVectorLike& values_in, DVVectorLike& keys_out, DVVectorLike& values_out, const Functor& binary_pred, size_t begin_key_in = 0, size_t end_key_in = (size_t)(-1), size_t begin_value_in = 0, size_t begin_key_out = 0, size_t begin_value_out = 0);


#endif