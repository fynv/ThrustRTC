#ifndef _TRTC_copy_h
#define _TRTC_copy_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out);

uint32_t THRUST_RTC_API TRTC_Copy_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred);
uint32_t THRUST_RTC_API TRTC_Copy_If_Stencil(const DVVectorLike& vec_in, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& pred);

#endif
