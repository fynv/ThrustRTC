#ifndef _TRTC_transform_h
#define _TRTC_transform_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Transform(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& op);
bool THRUST_RTC_API TRTC_Transform_Binary(const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, DVVectorLike& vec_out, const Functor& op);
bool THRUST_RTC_API TRTC_Transform_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& op, const Functor& pred);
bool THRUST_RTC_API TRTC_Transform_If_Stencil(const DVVectorLike& vec_in, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& op, const Functor& pred);
bool THRUST_RTC_API TRTC_Transform_Binary_If_Stencil(const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& op, const Functor& pred);

#endif
