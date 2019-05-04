#ifndef _TRTC_scan_h
#define _TRTC_scan_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Inclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);
bool THRUST_RTC_API TRTC_Inclusive_Scan(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_op, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);


#endif