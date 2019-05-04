#ifndef _TRTC_logical_h
#define _TRTC_logical_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_All_Of(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, bool& ret, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Any_Of(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, bool& ret, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_None_Of(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, bool& ret, size_t begin = 0, size_t end = (size_t)(-1));

#endif
