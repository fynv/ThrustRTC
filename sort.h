#ifndef _TRTC_sort_h
#define _TRTC_sort_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

// all merge-based
bool THRUST_RTC_API TRTC_Sort(DVVectorLike& vec, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Sort(DVVectorLike& vec, const Functor& comp, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Sort_By_Key(DVVectorLike& keys, DVVectorLike& values, size_t begin_keys = 0, size_t end_keys = (size_t)(-1), size_t begin_values = 0);
bool THRUST_RTC_API TRTC_Sort_By_Key(DVVectorLike& keys, DVVectorLike& values, const Functor& comp, size_t begin_keys = 0, size_t end_keys = (size_t)(-1), size_t begin_values = 0);

bool THRUST_RTC_API TRTC_Is_Sorted(const DVVectorLike& vec, bool& result, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Is_Sorted(const DVVectorLike& vec, const Functor& comp, bool& result, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Is_Sorted_Until(const DVVectorLike& vec, size_t& result, size_t begin = 0, size_t end = (size_t)(-1));
bool THRUST_RTC_API TRTC_Is_Sorted_Until(const DVVectorLike& vec, const Functor& comp, size_t& result, size_t begin = 0, size_t end = (size_t)(-1));


#endif 
