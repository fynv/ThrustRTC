#ifndef _TRTC_reduce_h
#define _TRTC_reduce_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Reduce(const DVVectorLike& vec, ViewBuf& ret);
bool THRUST_RTC_API TRTC_Reduce(const DVVectorLike& vec, const DeviceViewable& init, ViewBuf& ret);
bool THRUST_RTC_API TRTC_Reduce(const DVVectorLike& vec, const DeviceViewable& init, const Functor& binary_op, ViewBuf& ret);

uint32_t THRUST_RTC_API TRTC_Reduce_By_Key(const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out);
uint32_t THRUST_RTC_API TRTC_Reduce_By_Key(const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out, const Functor& binary_pred);
uint32_t THRUST_RTC_API TRTC_Reduce_By_Key(const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out, const Functor& binary_pred, const Functor& binary_op);

#endif
