#ifndef _TRTC_find_h
#define _TRTC_find_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Find(const DVVectorLike& vec, const DeviceViewable& value, size_t& result);
bool THRUST_RTC_API TRTC_Find_If(const DVVectorLike& vec, const Functor& pred, size_t& result);
bool THRUST_RTC_API TRTC_Find_If_Not(const DVVectorLike& vec, const Functor& pred, size_t& result);

#endif
