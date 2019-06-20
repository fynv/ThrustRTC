#ifndef _TRTC_transform_reduce_h
#define _TRTC_transform_reduce_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Transform_Reduce(const DVVectorLike& vec, const Functor& unary_op, const DeviceViewable& init, const Functor& binary_op, ViewBuf& ret);


#endif
