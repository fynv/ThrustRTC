#ifndef _TRTC_sequence_h
#define _TRTC_sequence_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"

bool THRUST_RTC_API TRTC_Sequence(DVVectorLike& vec);
bool THRUST_RTC_API TRTC_Sequence(DVVectorLike& vec, const DeviceViewable& value_init);
bool THRUST_RTC_API TRTC_Sequence(DVVectorLike& vec, const DeviceViewable& value_init, const DeviceViewable& value_step);

#endif
