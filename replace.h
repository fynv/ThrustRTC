#ifndef _TRTC_replace_h
#define _TRTC_replace_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Replace(TRTCContext& ctx, DVVector& vec, const DeviceViewable& old_value, const DeviceViewable& new_value, size_t begin = 0, size_t end = (size_t)(-1));
void THRUST_RTC_API TRTC_Replace_If(TRTCContext& ctx, DVVector& vec, const Functor& pred, const DeviceViewable& new_value, size_t begin = 0, size_t end = (size_t)(-1));

#endif

