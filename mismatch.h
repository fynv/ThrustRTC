#ifndef _TRTC_mismatch_h
#define _TRTC_mismatch_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Mismatch(const DVVectorLike& vec1, const DVVectorLike& vec2, size_t& result);
bool THRUST_RTC_API TRTC_Mismatch(const DVVectorLike& vec1, const DVVectorLike& vec2, const Functor& pred, size_t& result);


#endif
