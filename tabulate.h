#ifndef _TRTC_tabulate_h
#define _TRTC_tabulate_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

void THRUST_RTC_API TRTC_tabulate(TRTCContext& ctx, DVVector& vec, const Functor& op, size_t begin = 0, size_t end = (size_t)(-1));

#endif

