#ifndef _TRTC_scatter_h
#define _TRTC_scatter_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Scatter(const DVVectorLike& vec_in, const DVVectorLike& vec_map, DVVectorLike& vec_out);
bool THRUST_RTC_API TRTC_Scatter_If(const DVVectorLike& vec_in, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, DVVectorLike& vec_out);
bool THRUST_RTC_API TRTC_Scatter_If(const DVVectorLike& vec_in, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& pred);

#endif

