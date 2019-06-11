#include "launch_calc.h"
#include "cuda_occupancy.h"
#include <unordered_map>

static void s_get_occ_device_properties(cudaOccDeviceProp &occ_prop)
{
	CUdevice cuDevice;
	cuCtxGetDevice(&cuDevice);

	static std::unordered_map<CUdevice, cudaOccDeviceProp> s_dev_pro_map;
	decltype(s_dev_pro_map)::iterator it = s_dev_pro_map.find(cuDevice);
	if (it != s_dev_pro_map.end())
	{
		occ_prop = it->second;
		return;
	}

	cuDeviceGetAttribute(&occ_prop.computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
	cuDeviceGetAttribute(&occ_prop.computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
	cuDeviceGetAttribute(&occ_prop.maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cuDevice);
	cuDeviceGetAttribute(&occ_prop.maxThreadsPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cuDevice);
	cuDeviceGetAttribute(&occ_prop.regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, cuDevice);
	cuDeviceGetAttribute(&occ_prop.regsPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, cuDevice);
	cuDeviceGetAttribute(&occ_prop.warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevice);
	int i32value;
	cuDeviceGetAttribute(&i32value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuDevice);
	occ_prop.sharedMemPerBlock = (size_t)i32value;
	cuDeviceGetAttribute(&i32value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, cuDevice);
	occ_prop.sharedMemPerMultiprocessor = (size_t)i32value;
	cuDeviceGetAttribute(&occ_prop.numSms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice);
	s_dev_pro_map[cuDevice] = occ_prop;
	return;
}

static void s_get_occ_func_attributes(cudaOccFuncAttributes &occ_attrib, CUfunction func)
{
	cuFuncGetAttribute(&occ_attrib.maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func);
	cuFuncGetAttribute(&occ_attrib.numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
	int i32value;
	cuFuncGetAttribute(&i32value, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func);
	occ_attrib.sharedSizeBytes = (size_t)i32value;
	occ_attrib.partitionedGCConfig = PARTITIONED_GC_OFF;
	occ_attrib.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
	cuFuncGetAttribute(&i32value, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, func);
	occ_attrib.maxDynamicSharedSizeBytes = (size_t)i32value;
}

void launch_calc(CUfunction func, unsigned sharedMemBytes, int& sizeBlock)
{
	cudaOccDeviceProp occ_prop;
	s_get_occ_device_properties(occ_prop);
	cudaOccFuncAttributes occ_attrib;
	s_get_occ_func_attributes(occ_attrib, func);

	CUfunc_cache cacheConfig;
	cuCtxGetCacheConfig(&cacheConfig);

	cudaOccDeviceState occ_state;
	occ_state.cacheConfig = (cudaOccCacheConfig)cacheConfig;

	int min_grid_size = 0;
	cudaOccMaxPotentialOccupancyBlockSize(&min_grid_size,
		&sizeBlock,
		&occ_prop,
		&occ_attrib,
		&occ_state,
		0, size_t(sharedMemBytes));	
}

void persist_calc(CUfunction func, unsigned sharedMemBytes, int sizeBlock, int& numBlocks)
{
	cudaOccDeviceProp occ_prop;
	s_get_occ_device_properties(occ_prop);
	cudaOccFuncAttributes occ_attrib;
	s_get_occ_func_attributes(occ_attrib, func);

	CUfunc_cache cacheConfig;
	cuCtxGetCacheConfig(&cacheConfig);

	cudaOccDeviceState occ_state;
	occ_state.cacheConfig = (cudaOccCacheConfig)cacheConfig;

	cudaOccResult result;
	cudaOccMaxActiveBlocksPerMultiprocessor(&result, &occ_prop, &occ_attrib, &occ_state, sizeBlock, (size_t)sharedMemBytes);
	   
	numBlocks = result.activeBlocksPerMultiprocessor * occ_prop.numSms;
}
