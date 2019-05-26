#ifndef _TRTC_mismatch_h
#define _TRTC_mismatch_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Mismatch(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, size_t& result1, size_t& result2, size_t begin1 = 0, size_t end1 = (size_t)(-1), size_t begin2 = 0);
bool THRUST_RTC_API TRTC_Mismatch(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, const Functor& pred, size_t& result1, size_t& result2, size_t begin1 = 0, size_t end1 = (size_t)(-1), size_t begin2 = 0);


#endif
