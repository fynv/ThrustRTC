#ifndef _merge_sort_h
#define _merge_sort_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool merge_sort(TRTCContext& ctx, DVVectorLike& vec, const Functor& comp, size_t begin = 0, size_t end = (size_t)(-1));
bool merge_sort_by_key(TRTCContext& ctx, DVVectorLike& keys, DVVectorLike& values, const Functor& comp, size_t begin_keys = 0, size_t end_keys = (size_t)(-1), size_t begin_values = 0);


#endif
