#ifndef _TRTC_find_h
#define _TRTC_find_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Find(TRTCContext& ctx, const DVVectorLike& vec, const DeviceViewable& value, size_t& result, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Find_If(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, size_t& result, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Find_If_Not(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, size_t& result, size_t begin = 0, size_t end = (size_t)(-1));

#endif
