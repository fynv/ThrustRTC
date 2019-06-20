#ifndef _TRTC_unique_h
#define _TRTC_unique_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

uint32_t THRUST_RTC_API TRTC_Unique(DVVectorLike& vec);
uint32_t THRUST_RTC_API TRTC_Unique(DVVectorLike& vec, const Functor& binary_pred);
uint32_t THRUST_RTC_API TRTC_Unique_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out);
uint32_t THRUST_RTC_API TRTC_Unique_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_pred);
uint32_t THRUST_RTC_API TRTC_Unique_By_Key(DVVectorLike& keys, DVVectorLike& values);
uint32_t THRUST_RTC_API TRTC_Unique_By_Key(DVVectorLike& keys, DVVectorLike& values, const Functor& binary_pred);
uint32_t THRUST_RTC_API TRTC_Unique_By_Key_Copy(const DVVectorLike& keys_in, const DVVectorLike& values_in, DVVectorLike& keys_out, DVVectorLike& values_out);
uint32_t THRUST_RTC_API TRTC_Unique_By_Key_Copy(const DVVectorLike& keys_in, const DVVectorLike& values_in, DVVectorLike& keys_out, DVVectorLike& values_out, const Functor& binary_pred);


#endif