#ifndef _general_reduce_h
#define _general_reduce_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool general_reduce(size_t n, const char* name_cls, const Functor& src, const Functor& op, ViewBuf& ret_buf);


#endif
