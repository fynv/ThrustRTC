#ifndef _TRTC_for_each_h
#define _TRTC_for_each_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_For_Each(DVVectorLike& vec, const Functor& f);

#endif
