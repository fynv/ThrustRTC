#ifndef _general_scan_by_key_h
#define _general_scan_by_key_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool general_scan_by_key(TRTCContext& ctx, size_t n, const Functor& value_in, const DVVectorLike& key, DVVectorLike& value_out, const Functor& binary_pred, const Functor& binary_op, size_t begin_key, size_t begin_out);

#endif

