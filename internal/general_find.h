#ifndef _general_find_h
#define _general_find_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"


bool general_find(size_t begin, size_t end, const Functor src, size_t& result);


#endif