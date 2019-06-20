#ifndef _TRTC_scan_h
#define _TRTC_scan_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool THRUST_RTC_API TRTC_Inclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out);
bool THRUST_RTC_API TRTC_Inclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out, const Functor& binary_op);
bool THRUST_RTC_API TRTC_Exclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out);
bool THRUST_RTC_API TRTC_Exclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& init);
bool THRUST_RTC_API TRTC_Exclusive_Scan(const DVVectorLike& vec_in, DVVectorLike& vec_out, const DeviceViewable& init, const Functor& binary_op);

bool THRUST_RTC_API TRTC_Inclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out);
bool THRUST_RTC_API TRTC_Inclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const Functor& binary_pred);
bool THRUST_RTC_API TRTC_Inclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const Functor& binary_pred, const Functor& binary_op);
bool THRUST_RTC_API TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out);
bool THRUST_RTC_API TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const DeviceViewable& init);
bool THRUST_RTC_API TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const DeviceViewable& init, const Functor& binary_pred);
bool THRUST_RTC_API TRTC_Exclusive_Scan_By_Key(const DVVectorLike& vec_key, const DVVectorLike& vec_value, DVVectorLike& vec_out, const DeviceViewable& init, const Functor& binary_pred, const Functor& binary_op);


#endif