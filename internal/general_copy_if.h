#ifndef _general_copy_if_h
#define _general_copy_if_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

uint32_t general_copy_if(size_t n, const Functor& src_scan, const DVVectorLike& vec_in, DVVectorLike& vec_out);
uint32_t general_copy_if(size_t n, const Functor& src_scan, const DVVectorLike& vec_in1, const DVVectorLike& vec_in2, DVVectorLike& vec_out1, DVVectorLike& vec_out2);


#endif

