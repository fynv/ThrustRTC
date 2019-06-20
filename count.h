#ifndef _TRTC_count_h
#define _TRTC_count_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Count(const DVVectorLike& vec, const DeviceViewable& value, size_t& ret);
bool THRUST_RTC_API TRTC_Count_If(const DVVectorLike& vec, const Functor& pred, size_t& ret);

#endif
