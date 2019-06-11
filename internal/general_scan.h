#ifndef _general_scan_h
#define _general_scan_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool general_scan(size_t n, const Functor& src, DVVectorLike& vec_out, const Functor& binary_op, size_t begin_out);

#endif
