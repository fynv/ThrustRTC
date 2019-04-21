#ifndef _TRTC_count_h
#define _TRTC_count_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

size_t THRUST_RTC_API TRTC_Count(TRTCContext& ctx, const DVVectorLike& vec, const DeviceViewable& value, size_t begin = 0, size_t end = (size_t)(-1));
size_t THRUST_RTC_API TRTC_Count_If(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, size_t begin = 0, size_t end = (size_t)(-1));

#endif
