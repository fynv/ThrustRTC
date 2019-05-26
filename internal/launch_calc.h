#ifndef _launch_calc_h
#define _launch_calc_h

#include "cuda_wrapper.h"
void launch_calc(int dev_id, CUfunction func, unsigned sharedMemBytes, int& sizeBlock);
void persist_calc(int dev_id, CUfunction func, unsigned sharedMemBytes, int sizeBlock, int& numBlocks);

#endif
