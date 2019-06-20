#ifndef _TRTC_extrema_h
#define _TRTC_extrema_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Min_Element(const DVVectorLike& vec, size_t& id_min);
bool THRUST_RTC_API TRTC_Min_Element(const DVVectorLike& vec, const Functor& comp, size_t& id_min);
bool THRUST_RTC_API TRTC_Max_Element(const DVVectorLike& vec, size_t& id_max);
bool THRUST_RTC_API TRTC_Max_Element(const DVVectorLike& vec, const Functor& comp, size_t& id_max);
bool THRUST_RTC_API TRTC_MinMax_Element(const DVVectorLike& vec, size_t& id_min, size_t& id_max);
bool THRUST_RTC_API TRTC_MinMax_Element(const DVVectorLike& vec, const Functor& comp, size_t& id_min, size_t& id_max);


#endif
