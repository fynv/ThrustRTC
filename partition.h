#ifndef _TRTC_partition_h
#define _TRTC_partition_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

uint32_t THRUST_RTC_API TRTC_Partition(DVVectorLike& vec, const Functor& pred);
uint32_t THRUST_RTC_API TRTC_Partition_Stencil(DVVectorLike& vec, const DVVectorLike& stencil, const Functor& pred);
uint32_t THRUST_RTC_API TRTC_Partition_Copy(const DVVectorLike& vec_in, DVVectorLike& vec_true, DVVectorLike& vec_false, const Functor& pred);
uint32_t THRUST_RTC_API TRTC_Partition_Copy_Stencil(const DVVectorLike& vec_in, const DVVectorLike& stencil, DVVectorLike& vec_true, DVVectorLike& vec_false, const Functor& pred);

bool THRUST_RTC_API TRTC_Partition_Point(const DVVectorLike& vec, const Functor& pred, size_t& result);
bool THRUST_RTC_API TRTC_Is_Partitioned(const DVVectorLike& vec, const Functor& pred, bool& result);

#endif
