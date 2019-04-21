#ifndef _TRTC_copy_h
#define _TRTC_copy_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"

bool THRUST_RTC_API TRTC_Copy(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);


#endif
