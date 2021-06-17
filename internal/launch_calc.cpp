#include "TRTC_api.h"
#include "launch_calc.h"
#include "cuda_occupancy.h"
#include <unordered_map>

inline bool CheckCUresult(CUresult res, const char* name_call)
{
	if (res != CUDA_SUCCESS)
	{
		printf("%s failed with Error code: %u\n", name_call, res);
		const char *name = nullptr;
		const char *desc = nullptr;
		cuGetErrorName(res, &name);
		cuGetErrorString(res, &desc);
		if (name != nullptr)
		{
			printf("Error Name: %s\n", name);
		}
		if (desc != nullptr)
		{
			printf("Error Description: %s\n", desc);
		}
		return false;
	}
	return true;
}


static bool s_get_occ_device_properties(cudaOccDeviceProp &occ_prop)
{
	CUdevice cuDevice;
	if (!CheckCUresult(cuCtxGetDevice(&cuDevice), "cuCtxGetDevice()")) return false;

	static std::unordered_map<CUdevice, cudaOccDeviceProp> s_dev_pro_map;
	decltype(s_dev_pro_map)::iterator it = s_dev_pro_map.find(cuDevice);
	if (it != s_dev_pro_map.end())
	{
		occ_prop = it->second;
		return true;
	}

	if (!CheckCUresult(cuDeviceGetAttribute(&occ_prop.computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice), "cuDeviceGetAttribute()")) return false;
	if (!CheckCUresult(cuDeviceGetAttribute(&occ_prop.computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice), "cuDeviceGetAttribute()")) return false;
	if (!CheckCUresult(cuDeviceGetAttribute(&occ_prop.maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cuDevice), "cuDeviceGetAttribute()")) return false;
	if (!CheckCUresult(cuDeviceGetAttribute(&occ_prop.maxThreadsPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cuDevice), "cuDeviceGetAttribute()")) return false;
	if (!CheckCUresult(cuDeviceGetAttribute(&occ_prop.regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, cuDevice), "cuDeviceGetAttribute()")) return false;
	if (!CheckCUresult(cuDeviceGetAttribute(&occ_prop.regsPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, cuDevice), "cuDeviceGetAttribute()")) return false;
	if (!CheckCUresult(cuDeviceGetAttribute(&occ_prop.warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevice), "cuDeviceGetAttribute()")) return false;
	int i32value;
	if (!CheckCUresult(cuDeviceGetAttribute(&i32value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuDevice), "cuDeviceGetAttribute()")) return false;
	occ_prop.sharedMemPerBlock = (size_t)i32value;
	if (!CheckCUresult(cuDeviceGetAttribute(&i32value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, cuDevice), "cuDeviceGetAttribute()")) return false;
	occ_prop.sharedMemPerMultiprocessor = (size_t)i32value;
	if (!CheckCUresult(cuDeviceGetAttribute(&occ_prop.numSms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice), "cuDeviceGetAttribute()")) return false;
	s_dev_pro_map[cuDevice] = occ_prop;
	return true;
}

static bool s_get_occ_func_attributes(cudaOccFuncAttributes &occ_attrib, CUfunction func)
{
	if (!CheckCUresult(cuFuncGetAttribute(&occ_attrib.maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func), "cuFuncGetAttribute()")) return false;
	if (!CheckCUresult(cuFuncGetAttribute(&occ_attrib.numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, func), "cuFuncGetAttribute()")) return false;
	int i32value;
	if (!CheckCUresult(cuFuncGetAttribute(&i32value, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func), "cuFuncGetAttribute()")) return false;
	occ_attrib.sharedSizeBytes = (size_t)i32value;
	occ_attrib.partitionedGCConfig = PARTITIONED_GC_OFF;
	occ_attrib.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
	if (!CheckCUresult(cuFuncGetAttribute(&i32value, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, func), "cuFuncGetAttribute()")) return false;
	occ_attrib.maxDynamicSharedSizeBytes = (size_t)i32value;
	return true;
}

bool launch_calc(CUfunction func, unsigned sharedMemBytes, int& sizeBlock)
{
	cudaOccDeviceProp occ_prop;
	if (!s_get_occ_device_properties(occ_prop)) return false;
	cudaOccFuncAttributes occ_attrib;
	if (!s_get_occ_func_attributes(occ_attrib, func)) return false;

	CUfunc_cache cacheConfig;
	if (!CheckCUresult(cuCtxGetCacheConfig(&cacheConfig), "cuCtxGetCacheConfig()")) return false;

	cudaOccDeviceState occ_state;
	occ_state.cacheConfig = (cudaOccCacheConfig)cacheConfig;

	int min_grid_size = 0;
	cudaOccMaxPotentialOccupancyBlockSize(&min_grid_size,
		&sizeBlock,
		&occ_prop,
		&occ_attrib,
		&occ_state,
		0, size_t(sharedMemBytes));	

	return true;
}

bool persist_calc(CUfunction func, unsigned sharedMemBytes, int sizeBlock, int& numBlocks)
{
	cudaOccDeviceProp occ_prop;
	if (!s_get_occ_device_properties(occ_prop)) return false;
	cudaOccFuncAttributes occ_attrib;
	if (!s_get_occ_func_attributes(occ_attrib, func)) return false;

	CUfunc_cache cacheConfig;
	if (!CheckCUresult(cuCtxGetCacheConfig(&cacheConfig), "cuCtxGetCacheConfig()")) return false;

	cudaOccDeviceState occ_state;
	occ_state.cacheConfig = (cudaOccCacheConfig)cacheConfig;

	cudaOccResult result;
	cudaOccMaxActiveBlocksPerMultiprocessor(&result, &occ_prop, &occ_attrib, &occ_state, sizeBlock, (size_t)sharedMemBytes);
	   
	numBlocks = result.activeBlocksPerMultiprocessor * occ_prop.numSms;
	return true;
}
