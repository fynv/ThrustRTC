#ifndef _launch_calc_h
#define _launch_calc_h

#include "cuda_wrapper.h"
bool launch_calc(CUfunction func, unsigned sharedMemBytes, int& sizeBlock);
bool persist_calc(CUfunction func, unsigned sharedMemBytes, int sizeBlock, int& numBlocks);

#endif
