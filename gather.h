#ifndef _TRTC_gather_h
#define _TRTC_gather_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Gather(const DVVectorLike& vec_map, const DVVectorLike& vec_in, DVVectorLike& vec_out);
bool THRUST_RTC_API TRTC_Gather_If(const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, const DVVectorLike& vec_in, DVVectorLike& vec_out);
bool THRUST_RTC_API TRTC_Gather_If(const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred);

#endif

