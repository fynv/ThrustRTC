#ifndef _radix_sort_h
#define _radix_sort_h

#include "TRTC_api.h"
#include "TRTCContext.h"
#include "DeviceViewable.h"
#include "DVVector.h"
#include "functor.h"

bool radix_sort_32(DVVectorLike& vec, size_t begin = 0, size_t end = (size_t)(-1));
bool radix_sort_reverse_32(DVVectorLike& vec, size_t begin = 0, size_t end = (size_t)(-1));
bool radix_sort_64(DVVectorLike& vec, size_t begin = 0, size_t end = (size_t)(-1));
bool radix_sort_reverse_64(DVVectorLike& vec, size_t begin = 0, size_t end = (size_t)(-1));

bool radix_sort_by_key_32(DVVectorLike& keys, DVVectorLike& values, size_t begin_keys, size_t end_keys, size_t begin_values);
bool radix_sort_by_key_reverse_32(DVVectorLike& keys, DVVectorLike& values, size_t begin_keys, size_t end_keys, size_t begin_values);
bool radix_sort_by_key_64(DVVectorLike& keys, DVVectorLike& values, size_t begin_keys, size_t end_keys, size_t begin_values);
bool radix_sort_by_key_reverse_64(DVVectorLike& keys, DVVectorLike& values, size_t begin_keys, size_t end_keys, size_t begin_values);


#endif
