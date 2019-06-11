#ifndef _TRTC_replace_h
#define _TRTC_replace_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Replace(DVVectorLike& vec, const DeviceViewable& old_value, const DeviceViewable& new_value, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Replace_If(DVVectorLike& vec, const Functor& pred, const DeviceViewable& new_value, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Replace_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& old_value, const DeviceViewable& new_value, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);
bool THRUST_RTC_API TRTC_Replace_Copy_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred, const DeviceViewable& new_value, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);

#endif

