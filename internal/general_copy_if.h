#ifndef _general_copy_if_h
#define _general_copy_if_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

uint32_t general_copy_if(TRTCContext& ctx, size_t n, const Functor& src_scan, const DVVectorLike& vec_in, DVVectorLike& vec_out,  size_t begin_in = 0, size_t begin_out = 0);


#endif

