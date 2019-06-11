#ifndef _TRTC_copy_h
#define _TRTC_copy_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);

uint32_t THRUST_RTC_API TRTC_Copy_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);
uint32_t THRUST_RTC_API TRTC_Copy_If_Stencil(const DVVectorLike& vec_in, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_stencil=0, size_t begin_out = 0);

#endif
