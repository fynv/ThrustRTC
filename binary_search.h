#ifndef _TRTC_binary_search_h
#define _TRTC_binary_search_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Lower_Bound(TRTCContext& ctx, const DVVectorLike& vec, const DeviceViewable& value, size_t& result, size_t begin = 0, size_t end = (size_t)(-1));


#endif