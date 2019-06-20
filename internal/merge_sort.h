#ifndef _merge_sort_h
#define _merge_sort_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool merge_sort(DVVectorLike& vec, const Functor& comp);
bool merge_sort_by_key(DVVectorLike& keys, DVVectorLike& values, const Functor& comp);


#endif
