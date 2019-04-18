#ifndef _TRTC_transform_h
#define _TRTC_transform_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

void THRUST_RTC_API TRTC_Transform(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, const Functor& op, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);
void THRUST_RTC_API TRTC_Transform_Binary(TRTCContext& ctx, const DVVector& vec_in1, const DVVector& vec_in2, DVVector& vec_out, const Functor& op, size_t begin_in1 = 0, size_t end_in1 = (size_t)(-1), size_t begin_in2 = 0, size_t begin_out = 0);
void THRUST_RTC_API TRTC_Transform_If(TRTCContext& ctx, const DVVector& vec_in, DVVector& vec_out, const Functor& op, const Functor& pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);
void THRUST_RTC_API TRTC_Transform_If_Stencil(TRTCContext& ctx, const DVVector& vec_in, const DVVector& vec_stencil, DVVector& vec_out, const Functor& op, const Functor& pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_stencil = 0, size_t begin_out = 0);
void THRUST_RTC_API TRTC_Transform_Binary_If_Stencil(TRTCContext& ctx, const DVVector& vec_in1, const DVVector& vec_in2, const DVVector& vec_stencil, DVVector& vec_out, const Functor& op, const Functor& pred, size_t begin_in1 = 0, size_t end_in1 = (size_t)(-1), size_t begin_in2 = 0, size_t begin_stencil = 0, size_t begin_out = 0);

#endif
