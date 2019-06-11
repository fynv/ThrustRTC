#ifndef _TRTC_transform_h
#define _TRTC_transform_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Transform(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& op, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);
bool THRUST_RTC_API TRTC_Transform_Binary(const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, DVVectorLike& vec_out, const Functor& op, size_t begin_in1 = 0, size_t end_in1 = (size_t)(-1), size_t begin_in2 = 0, size_t begin_out = 0);
bool THRUST_RTC_API TRTC_Transform_If(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& op, const Functor& pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_out = 0);
bool THRUST_RTC_API TRTC_Transform_If_Stencil(const DVVectorLike& vec_in, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& op, const Functor& pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_stencil = 0, size_t begin_out = 0);
bool THRUST_RTC_API TRTC_Transform_Binary_If_Stencil(const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, const DVVectorLike& vec_stencil, DVVectorLike& vec_out, const Functor& op, const Functor& pred, size_t begin_in1 = 0, size_t end_in1 = (size_t)(-1), size_t begin_in2 = 0, size_t begin_stencil = 0, size_t begin_out = 0);

#endif
