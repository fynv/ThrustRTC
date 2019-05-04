#ifndef _TRTC_inner_product_h
#define _TRTC_inner_product_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Inner_Product(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, const DeviceViewable& init, ViewBuf& ret, size_t begin1 = 0, size_t end1 = (size_t)(-1), size_t begin2 = 0);
bool THRUST_RTC_API TRTC_Inner_Product(TRTCContext& ctx, const DVVectorLike& vec1, const DVVectorLike& vec2, const DeviceViewable& init, ViewBuf& ret, const Functor& binary_op1, const Functor& binary_op2, size_t begin1 = 0, size_t end1 = (size_t)(-1), size_t begin2 = 0);


#endif
