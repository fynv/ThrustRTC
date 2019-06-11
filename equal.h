#ifndef _TRTC_equal_h
#define _TRTC_equal_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Equal(const DVVectorLike& vec1, const DVVectorLike& vec2, bool& ret, size_t begin1 = 0, size_t end1 = (size_t)(-1), size_t begin2 = 0);
bool THRUST_RTC_API TRTC_Equal(const DVVectorLike& vec1, const DVVectorLike& vec2, const Functor& binary_pred, bool& ret, size_t begin1 = 0, size_t end1 = (size_t)(-1), size_t begin2 = 0);

#endif
