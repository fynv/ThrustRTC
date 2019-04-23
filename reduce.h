#ifndef _TRTC_reduce_h
#define _TRTC_reduce_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Reduce(TRTCContext& ctx, const DVVectorLike& vec, ViewBuf& ret, size_t begin = 0, size_t end = (size_t)(-1));


#endif
