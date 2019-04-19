#ifndef _TRTC_fill_h
#define _TRTC_fill_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"

bool THRUST_RTC_API TRTC_Fill(TRTCContext& ctx, DVVectorLike& vec, const DeviceViewable& value, size_t begin = 0, size_t end = (size_t)(-1));


#endif
