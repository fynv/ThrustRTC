#ifndef _TRTC_remove_h
#define _TRTC_remove_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

uint32_t THRUST_RTC_API TRTC_Remove(DVVectorLike& vec, const DeviceViewable& value, size_t begin = 0, size_t end = (size_t)(-1));
uint32_t THRUST_RTC_API TRTC_Remove_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& value, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);
uint32_t THRUST_RTC_API TRTC_Remove_If(DVVectorLike& vec, const Functor& pred, size_t begin = 0, size_t end = (size_t)(-1));
uint32_t THRUST_RTC_API TRTC_Remove_Copy_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);
uint32_t THRUST_RTC_API TRTC_Remove_If_Stencil(DVVectorLike& vec, const DVVectorLike& stencil, const Functor& pred, size_t begin = 0, size_t end = (size_t)(-1), size_t begin_stencil = 0);
uint32_t THRUST_RTC_API TRTC_Remove_Copy_If_Stencil(const DVVectorLike& vec_in, const DVVectorLike& stencil, DVVectorLike& vec_out, const Functor& pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_stencil = 0, size_t begin_out = 0);


#endif