#ifndef _TRTC_sort_h
#define _TRTC_sort_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

// all merge-based
bool THRUST_RTC_API TRTC_Sort(DVVectorLike& vec);
bool THRUST_RTC_API TRTC_Sort(DVVectorLike& vec, const Functor& comp);
bool THRUST_RTC_API TRTC_Sort_By_Key(DVVectorLike& keys, DVVectorLike& values);
bool THRUST_RTC_API TRTC_Sort_By_Key(DVVectorLike& keys, DVVectorLike& values, const Functor& comp);

bool THRUST_RTC_API TRTC_Is_Sorted(const DVVectorLike& vec, bool& result);
bool THRUST_RTC_API TRTC_Is_Sorted(const DVVectorLike& vec, const Functor& comp, bool& result);
bool THRUST_RTC_API TRTC_Is_Sorted_Until(const DVVectorLike& vec, size_t& result);
bool THRUST_RTC_API TRTC_Is_Sorted_Until(const DVVectorLike& vec, const Functor& comp, size_t& result);


#endif 
