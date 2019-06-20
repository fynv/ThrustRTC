#ifndef _TRTC_binary_search_h
#define _TRTC_binary_search_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Lower_Bound(const DVVectorLike& vec, const DeviceViewable& value, size_t& result);
bool THRUST_RTC_API TRTC_Lower_Bound(const DVVectorLike& vec, const DeviceViewable& value, const Functor& comp, size_t& result);
bool THRUST_RTC_API TRTC_Upper_Bound(const DVVectorLike& vec, const DeviceViewable& value, size_t& result);
bool THRUST_RTC_API TRTC_Upper_Bound(const DVVectorLike& vec, const DeviceViewable& value, const Functor& comp, size_t& result);

bool THRUST_RTC_API TRTC_Binary_Search(const DVVectorLike& vec, const DeviceViewable& value, bool& result);
bool THRUST_RTC_API TRTC_Binary_Search(const DVVectorLike& vec, const DeviceViewable& value, const Functor& comp, bool& result);

bool THRUST_RTC_API TRTC_Lower_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result);
bool THRUST_RTC_API TRTC_Lower_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, const Functor& comp);

bool THRUST_RTC_API TRTC_Upper_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result);
bool THRUST_RTC_API TRTC_Upper_Bound_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, const Functor& comp);

bool THRUST_RTC_API TRTC_Binary_Search_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result);
bool THRUST_RTC_API TRTC_Binary_Search_V(const DVVectorLike& vec, const DVVectorLike& values, DVVectorLike& result, const Functor& comp);


#endif