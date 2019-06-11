#ifndef _TRTC_gather_h
#define _TRTC_gather_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Gather(const DVVectorLike& vec_map, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_map = 0, size_t end_map = (size_t)(-1), size_t begin_in=0, size_t begin_out=0);
bool THRUST_RTC_API TRTC_Gather_If(const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_map = 0, size_t end_map = (size_t)(-1), size_t begin_stencil = 0, size_t begin_in = 0, size_t begin_out = 0);
bool THRUST_RTC_API TRTC_Gather_If(const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred, size_t begin_map = 0, size_t end_map = (size_t)(-1), size_t begin_stencil = 0, size_t begin_in = 0, size_t begin_out = 0);

#endif

