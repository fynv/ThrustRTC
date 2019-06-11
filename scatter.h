#ifndef _TRTC_scatter_h
#define _TRTC_scatter_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Scatter(const DVVectorLike& vec_in, const DVVectorLike& vec_map, DVVectorLike& vec_out, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_map = 0, size_t begin_out = 0);
bool THRUST_RTC_API TRTC_Scatter_If(const DVVectorLike& vec_in, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_map = 0, size_t begin_stencil = 0, size_t begin_out = 0);
bool THRUST_RTC_API TRTC_Scatter_If(const DVVectorLike& vec_in, const DVVectorLike& vec_map, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_map = 0, size_t begin_stencil = 0, size_t begin_out = 0);

#endif

