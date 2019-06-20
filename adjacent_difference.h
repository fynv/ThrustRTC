#ifndef _TRTC_adjacent_difference_h
#define _TRTC_adjacent_difference_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Adjacent_Difference(const DVVectorLike& vec_in, DVVectorLike& vec_out);
bool THRUST_RTC_API TRTC_Adjacent_Difference(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_op);

#endif

