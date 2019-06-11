#ifndef _TRTC_swap_h
#define _TRTC_swap_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"

bool THRUST_RTC_API TRTC_Swap(DVVectorLike& vec1, DVVectorLike& vec2, size_t begin1 = 0, size_t end1 = (size_t)(-1), size_t begin2 = 0);


#endif
