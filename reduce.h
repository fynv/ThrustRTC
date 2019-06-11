#ifndef _TRTC_reduce_h
#define _TRTC_reduce_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Reduce(const DVVectorLike& vec, ViewBuf& ret, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Reduce(const DVVectorLike& vec, const DeviceViewable& init, ViewBuf& ret, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Reduce(const DVVectorLike& vec, const DeviceViewable& init, const Functor& binary_op, ViewBuf& ret, size_t begin = 0, size_t end = (size_t)(-1));

uint32_t THRUST_RTC_API TRTC_Reduce_By_Key(const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out, size_t begin_key_in = 0, size_t end_key_in = (size_t)(-1), size_t begin_value_in = 0, size_t begin_key_out = 0, size_t begin_value_out = 0);
uint32_t THRUST_RTC_API TRTC_Reduce_By_Key(const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out, const Functor& binary_pred, size_t begin_key_in = 0, size_t end_key_in = (size_t)(-1), size_t begin_value_in = 0, size_t begin_key_out = 0, size_t begin_value_out = 0);
uint32_t THRUST_RTC_API TRTC_Reduce_By_Key(const DVVectorLike& key_in, const DVVectorLike& value_in, DVVectorLike& key_out, DVVectorLike& value_out, const Functor& binary_pred, const Functor& binary_op, size_t begin_key_in = 0, size_t end_key_in = (size_t)(-1), size_t begin_value_in = 0, size_t begin_key_out = 0, size_t begin_value_out = 0);

#endif
