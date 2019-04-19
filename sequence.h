#ifndef _TRTC_sequence_h
#define _TRTC_sequence_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"

bool THRUST_RTC_API TRTC_Sequence(TRTCContext& ctx, DVVector& vec, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Sequence(TRTCContext& ctx, DVVector& vec, const DeviceViewable& value_init, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Sequence(TRTCContext& ctx, DVVector& vec, const DeviceViewable& value_init, const DeviceViewable& value_step, size_t begin = 0, size_t end = (size_t)(-1));

#endif
