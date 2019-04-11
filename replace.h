#ifndef TRTC_replace_h
#define TRTC_replace_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"

bool THRUST_RTC_API TRTC_Replace(TRTCContext& ctx, DVVector& vec, const DeviceViewable& old_value, const DeviceViewable& new_value, size_t begin = 0, size_t end = (size_t)(-1));


#endif

