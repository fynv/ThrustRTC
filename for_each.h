#ifndef _TRTC_for_each_h
#define _TRTC_for_each_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DVVector.h"
#include "functor.h"

void THRUST_RTC_API TRTC_For_Each(TRTCContext& ctx, DVVector& vec, const Functor& f, size_t begin = 0, size_t end = (size_t)(-1));

#endif
