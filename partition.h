#ifndef _TRTC_partition_h
#define _TRTC_partition_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

uint32_t THRUST_RTC_API TRTC_Partition(TRTCContext& ctx, DVVectorLike& vec, const Functor& pred, size_t begin = 0, size_t end = (size_t)(-1));
uint32_t THRUST_RTC_API TRTC_Partition_Stencil(TRTCContext& ctx, DVVectorLike& vec, const DVVectorLike& stencil, const Functor& pred, size_t begin = 0, size_t end = (size_t)(-1), size_t begin_stencil = 0);
uint32_t THRUST_RTC_API TRTC_Partition_Copy(TRTCContext& ctx, const DVVectorLike& vec_in, DVVectorLike& vec_true, DVVectorLike& vec_false, const Functor& pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_true = 0, size_t begin_false = 0);
uint32_t THRUST_RTC_API TRTC_Partition_Copy_Stencil(TRTCContext& ctx, const DVVectorLike& vec_in, const DVVectorLike& stencil, DVVectorLike& vec_true, DVVectorLike& vec_false, const Functor& pred, size_t begin_in = 0, size_t end_in = (size_t)(-1), size_t begin_stencil = 0, size_t begin_true = 0, size_t begin_false = 0);

bool THRUST_RTC_API TRTC_Partition_Point(TRTCContext& ctx, const DVVectorLike& vec, const Functor& pred, size_t& result, size_t begin = 0, size_t end = (size_t)(-1));


#endif
