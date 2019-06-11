#include "cuda_wrapper.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <dlfcn.h>
#endif

#include <stdio.h>

static bool s_cuda_initialized = false;

CUresult(*cuInit)(unsigned int Flags);
CUresult(*cuDeviceGetCount)(int *count);
CUresult(*cuDeviceGet)(CUdevice *device, int ordinal);
CUresult(*cuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult(*cuCtxCreate)(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult(*cuCtxGetCurrent)(CUcontext *pctx);
CUresult(*cuCtxGetDevice)(CUdevice *device);
CUresult(*cuCtxGetCacheConfig)(CUfunc_cache *pconfig);
CUresult(*cuModuleLoadDataEx)(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult(*cuModuleUnload)(CUmodule hmod);
CUresult(*cuModuleGetGlobal)(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
CUresult(*cuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult(*cuFuncGetAttribute)(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult(*cuMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
CUresult(*cuMemFree)(CUdeviceptr dptr);
CUresult(*cuMemsetD8)(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUresult(*cuMemcpyHtoD)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult(*cuMemcpyDtoH)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult(*cuLaunchKernel)(CUfunction f,
	unsigned int gridDimX,
	unsigned int gridDimY,
	unsigned int gridDimZ,
	unsigned int blockDimX,
	unsigned int blockDimY,
	unsigned int blockDimZ,
	unsigned int sharedMemBytes,
	CUstream hStream,
	void **kernelParams,
	void **extra);

bool init_cuda()
{
	if (s_cuda_initialized) return true;

#ifdef _WIN32
	HINSTANCE hinstLib = LoadLibraryA("nvcuda.dll");
	if (hinstLib == NULL)
	{
		printf("nvcuda.dll not found\n");
		return false;
	}
	cuInit = (decltype(cuInit))GetProcAddress(hinstLib, "cuInit");
	cuDeviceGetCount = (decltype(cuDeviceGetCount))GetProcAddress(hinstLib, "cuDeviceGetCount");
	cuDeviceGet = (decltype(cuDeviceGet))GetProcAddress(hinstLib, "cuDeviceGet");
	cuDeviceGetAttribute = (decltype(cuDeviceGetAttribute))GetProcAddress(hinstLib, "cuDeviceGetAttribute");
	cuCtxCreate = (decltype(cuCtxCreate))GetProcAddress(hinstLib, "cuCtxCreate_v2");
	cuCtxGetCurrent = (decltype(cuCtxGetCurrent))GetProcAddress(hinstLib, "cuCtxGetCurrent");
	cuCtxGetDevice = (decltype(cuCtxGetDevice))GetProcAddress(hinstLib, "cuCtxGetDevice");
	cuCtxGetCacheConfig = (decltype(cuCtxGetCacheConfig))GetProcAddress(hinstLib, "cuCtxGetCacheConfig");
	cuModuleLoadDataEx = (decltype(cuModuleLoadDataEx))GetProcAddress(hinstLib, "cuModuleLoadDataEx");
	cuModuleUnload = (decltype(cuModuleUnload))GetProcAddress(hinstLib, "cuModuleUnload");
	cuModuleGetGlobal = (decltype(cuModuleGetGlobal))GetProcAddress(hinstLib, "cuModuleGetGlobal_v2");
	cuModuleGetFunction = (decltype(cuModuleGetFunction))GetProcAddress(hinstLib, "cuModuleGetFunction");
	cuFuncGetAttribute = (decltype(cuFuncGetAttribute))GetProcAddress(hinstLib, "cuFuncGetAttribute");
	cuMemAlloc = (decltype(cuMemAlloc))GetProcAddress(hinstLib, "cuMemAlloc_v2");
	cuMemFree = (decltype(cuMemFree))GetProcAddress(hinstLib, "cuMemFree_v2");
	cuMemsetD8 = (decltype(cuMemsetD8))GetProcAddress(hinstLib, "cuMemsetD8_v2");
	cuMemcpyHtoD = (decltype(cuMemcpyHtoD))GetProcAddress(hinstLib, "cuMemcpyHtoD_v2");
	cuMemcpyDtoH = (decltype(cuMemcpyDtoH))GetProcAddress(hinstLib, "cuMemcpyDtoH_v2");
	cuLaunchKernel = (decltype(cuLaunchKernel))GetProcAddress(hinstLib, "cuLaunchKernel");
#else
	void *handle = dlopen("libcuda.so", RTLD_LAZY);
	if (!handle)
	{
		printf("libcuda.so not found\n");
		return false;
	}
	cuInit = (decltype(cuInit))dlsym(handle, "cuInit");
	cuDeviceGetCount = (decltype(cuDeviceGetCount))dlsym(handle, "cuDeviceGetCount");
	cuDeviceGet = (decltype(cuDeviceGet))dlsym(handle, "cuDeviceGet");
	cuDeviceGetAttribute = (decltype(cuDeviceGetAttribute))dlsym(handle, "cuDeviceGetAttribute");
	cuCtxCreate = (decltype(cuCtxCreate))dlsym(handle, "cuCtxCreate_v2");
	cuCtxGetCurrent = (decltype(cuCtxGetCurrent))dlsym(handle, "cuCtxGetCurrent");
	cuCtxGetDevice = (decltype(cuCtxGetDevice))dlsym(handle, "cuCtxGetDevice");
	cuCtxGetCacheConfig = (decltype(cuCtxGetCacheConfig))dlsym(handle, "cuCtxGetCacheConfig");
	cuModuleLoadDataEx = (decltype(cuModuleLoadDataEx))dlsym(handle, "cuModuleLoadDataEx");
	cuModuleUnload = (decltype(cuModuleUnload))dlsym(handle, "cuModuleUnload");
	cuModuleGetGlobal = (decltype(cuModuleGetGlobal))dlsym(handle, "cuModuleGetGlobal_v2");
	cuModuleGetFunction = (decltype(cuModuleGetFunction))dlsym(handle, "cuModuleGetFunction");
	cuFuncGetAttribute = (decltype(cuFuncGetAttribute))dlsym(handle, "cuFuncGetAttribute");
	cuMemAlloc = (decltype(cuMemAlloc))dlsym(handle, "cuMemAlloc_v2");
	cuMemFree = (decltype(cuMemFree))dlsym(handle, "cuMemFree_v2");
	cuMemsetD8 = (decltype(cuMemsetD8))dlsym(handle, "cuMemsetD8_v2");
	cuMemcpyHtoD = (decltype(cuMemcpyHtoD))dlsym(handle, "cuMemcpyHtoD_v2");
	cuMemcpyDtoH = (decltype(cuMemcpyDtoH))dlsym(handle, "cuMemcpyDtoH_v2");
	cuLaunchKernel = (decltype(cuLaunchKernel))dlsym(handle, "cuLaunchKernel");
#endif

	s_cuda_initialized = true;
	return true;
}
