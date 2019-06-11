#ifndef _cuda_wrapper_h
#define _cuda_wrapper_h

#include <cstddef>

typedef unsigned long long CUdeviceptr;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

typedef enum CUdevice_attribute_enum {
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,              /**< Maximum number of threads per block */
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,                    /**< Maximum block dimension X */
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,                    /**< Maximum block dimension Y */
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,                    /**< Maximum block dimension Z */
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,                     /**< Maximum grid dimension X */
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,                     /**< Maximum grid dimension Y */
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,                     /**< Maximum grid dimension Z */
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,        /**< Maximum shared memory available per block in bytes */
	CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,            /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
	CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,              /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
	CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,                         /**< Warp size in threads */
	CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,                         /**< Maximum pitch in bytes allowed by memory copies */
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,           /**< Maximum number of 32-bit registers available per block */
	CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,               /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
	CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,                        /**< Typical clock frequency in kilohertz */
	CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,                 /**< Alignment requirement for textures */
	CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,                       /**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
	CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,              /**< Number of multiprocessors on device */
	CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,               /**< Specifies whether there is a run time limit on kernels */
	CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,                        /**< Device is integrated with host memory */
	CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,               /**< Device can map host memory into CUDA address space */
	CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,                      /**< Compute mode (See ::CUcomputemode for details) */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,           /**< Maximum 1D texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,           /**< Maximum 2D texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,          /**< Maximum 2D texture height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,           /**< Maximum 3D texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,          /**< Maximum 3D texture height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,           /**< Maximum 3D texture depth */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,   /**< Maximum 2D layered texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,  /**< Maximum 2D layered texture height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,  /**< Maximum layers in a 2D layered texture */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,     /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,    /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29, /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
	CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,                 /**< Alignment requirement for surfaces */
	CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,                /**< Device can possibly execute multiple kernels concurrently */
	CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,                       /**< Device has ECC support enabled */
	CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,                        /**< PCI bus ID of the device */
	CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,                     /**< PCI device ID of the device */
	CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,                        /**< Device is using TCC driver model */
	CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,                 /**< Peak memory clock frequency in kilohertz */
	CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,           /**< Global memory bus width in bits */
	CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,                     /**< Size of L2 cache in bytes */
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,    /**< Maximum resident threads per multiprocessor */
	CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,                /**< Number of asynchronous engines */
	CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,                /**< Device shares a unified address space with the host */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,   /**< Maximum 1D layered texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,  /**< Maximum layers in a 1D layered texture */
	CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,                  /**< Deprecated, do not use. */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,    /**< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,   /**< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47, /**< Alternate maximum 3D texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,/**< Alternate maximum 3D texture height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49, /**< Alternate maximum 3D texture depth */
	CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,                     /**< PCI domain ID of the device */
	CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,           /**< Pitch alignment requirement for textures */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,      /**< Maximum cubemap texture width/height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,  /**< Maximum cubemap layered texture width/height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54, /**< Maximum layers in a cubemap layered texture */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,           /**< Maximum 1D surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,           /**< Maximum 2D surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,          /**< Maximum 2D surface height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,           /**< Maximum 3D surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,          /**< Maximum 3D surface height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,           /**< Maximum 3D surface depth */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,   /**< Maximum 1D layered surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,  /**< Maximum layers in a 1D layered surface */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,   /**< Maximum 2D layered surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,  /**< Maximum 2D layered surface height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,  /**< Maximum layers in a 2D layered surface */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,      /**< Maximum cubemap surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,  /**< Maximum cubemap layered surface width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68, /**< Maximum layers in a cubemap layered surface */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,    /**< Maximum 1D linear texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,    /**< Maximum 2D linear texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,   /**< Maximum 2D linear texture height */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,    /**< Maximum 2D linear texture pitch in bytes */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73, /**< Maximum mipmapped 2D texture width */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,/**< Maximum mipmapped 2D texture height */
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,          /**< Major compute capability version number */
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,          /**< Minor compute capability version number */
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77, /**< Maximum mipmapped 1D texture width */
	CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,       /**< Device supports stream priorities */
	CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,         /**< Device supports caching globals in L1 */
	CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,          /**< Device supports caching locals in L1 */
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,  /**< Maximum shared memory available per multiprocessor in bytes */
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,  /**< Maximum number of 32-bit registers available per multiprocessor */
	CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,                    /**< Device can allocate managed memory on this system */
	CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,                    /**< Device is on a multi-GPU board */
	CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,           /**< Unique id for a group of devices on the same multi-GPU board */
	CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,       /**< Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)*/
	CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,  /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
	CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,            /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
	CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,         /**< Device can coherently access managed memory concurrently with the CPU */
	CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,      /**< Device supports compute preemption. */
	CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,            /**< ::cuStreamBatchMemOp and related APIs are supported. */
	CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,     /**< 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs. */
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,     /**< ::CU_STREAM_WAIT_VALUE_NOR is supported. */
	CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,                /**< Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel */
	CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,   /**< Device can participate in cooperative kernels launched via ::cuLaunchCooperativeKernelMultiDevice */
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97, /**< Maximum optin shared memory per block */
	CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,           /**< Both the ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details. */
	CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,           /**< Device supports host memory registration via ::cudaHostRegister. */
	CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100, /**< Device accesses pageable memory via the host's page tables. */
	CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101, /**< The host can directly access managed memory on the device without migration. */
	CU_DEVICE_ATTRIBUTE_MAX
} CUdevice_attribute;

typedef enum CUfunc_cache_enum {
	CU_FUNC_CACHE_PREFER_NONE = 0x00, /**< no preference for shared memory or L1 (default) */
	CU_FUNC_CACHE_PREFER_SHARED = 0x01, /**< prefer larger shared memory and smaller L1 cache */
	CU_FUNC_CACHE_PREFER_L1 = 0x02, /**< prefer larger L1 cache and smaller shared memory */
	CU_FUNC_CACHE_PREFER_EQUAL = 0x03  /**< prefer equal sized L1 cache and shared memory */
} CUfunc_cache;

typedef enum CUjit_option_enum
{
	CU_JIT_MAX_REGISTERS = 0,
	CU_JIT_THREADS_PER_BLOCK,
	CU_JIT_WALL_TIME,
	CU_JIT_INFO_LOG_BUFFER,
	CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
	CU_JIT_ERROR_LOG_BUFFER,
	CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
	CU_JIT_OPTIMIZATION_LEVEL,
	CU_JIT_TARGET_FROM_CUCONTEXT,
	CU_JIT_TARGET,
	CU_JIT_FALLBACK_STRATEGY,
	CU_JIT_GENERATE_DEBUG_INFO,
	CU_JIT_LOG_VERBOSE,
	CU_JIT_GENERATE_LINE_INFO,
	CU_JIT_CACHE_MODE,
	CU_JIT_NEW_SM3X_OPT,
	CU_JIT_FAST_COMPILE,
	CU_JIT_GLOBAL_SYMBOL_NAMES,
	CU_JIT_GLOBAL_SYMBOL_ADDRESSES,
	CU_JIT_GLOBAL_SYMBOL_COUNT,
	CU_JIT_NUM_OPTIONS
} CUjit_option;

typedef enum CUfunction_attribute_enum {
	CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
	CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
	CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
	CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
	CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
	CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
	CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
	CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
	CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
	CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
	CU_FUNC_ATTRIBUTE_MAX
} CUfunction_attribute;

typedef enum cudaError_enum {
	CUDA_SUCCESS = 0,
	CUDA_ERROR_INVALID_VALUE = 1,
	CUDA_ERROR_OUT_OF_MEMORY = 2,
	CUDA_ERROR_NOT_INITIALIZED = 3,
	CUDA_ERROR_DEINITIALIZED = 4,
	CUDA_ERROR_PROFILER_DISABLED = 5,
	CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
	CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
	CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
	CUDA_ERROR_NO_DEVICE = 100,
	CUDA_ERROR_INVALID_DEVICE = 101,
	CUDA_ERROR_INVALID_IMAGE = 200,
	CUDA_ERROR_INVALID_CONTEXT = 201,
	CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
	CUDA_ERROR_MAP_FAILED = 205,
	CUDA_ERROR_UNMAP_FAILED = 206,
	CUDA_ERROR_ARRAY_IS_MAPPED = 207,
	CUDA_ERROR_ALREADY_MAPPED = 208,
	CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
	CUDA_ERROR_ALREADY_ACQUIRED = 210,
	CUDA_ERROR_NOT_MAPPED = 211,
	CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
	CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
	CUDA_ERROR_ECC_UNCORRECTABLE = 214,
	CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
	CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
	CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
	CUDA_ERROR_INVALID_PTX = 218,
	CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
	CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
	CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
	CUDA_ERROR_INVALID_SOURCE = 300,
	CUDA_ERROR_FILE_NOT_FOUND = 301,
	CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
	CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
	CUDA_ERROR_OPERATING_SYSTEM = 304,
	CUDA_ERROR_INVALID_HANDLE = 400,
	CUDA_ERROR_ILLEGAL_STATE = 401,
	CUDA_ERROR_NOT_FOUND = 500,
	CUDA_ERROR_NOT_READY = 600,
	CUDA_ERROR_ILLEGAL_ADDRESS = 700,
	CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
	CUDA_ERROR_LAUNCH_TIMEOUT = 702,
	CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
	CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
	CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
	CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
	CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
	CUDA_ERROR_ASSERT = 710,
	CUDA_ERROR_TOO_MANY_PEERS = 711,
	CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
	CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
	CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
	CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
	CUDA_ERROR_MISALIGNED_ADDRESS = 716,
	CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
	CUDA_ERROR_INVALID_PC = 718,
	CUDA_ERROR_LAUNCH_FAILED = 719,
	CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
	CUDA_ERROR_NOT_PERMITTED = 800,
	CUDA_ERROR_NOT_SUPPORTED = 801,
	CUDA_ERROR_SYSTEM_NOT_READY = 802,
	CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
	CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
	CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
	CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
	CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
	CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
	CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
	CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
	CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
	CUDA_ERROR_CAPTURED_EVENT = 907,
	CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
	CUDA_ERROR_UNKNOWN = 999
} CUresult;

extern CUresult (*cuInit)(unsigned int Flags);
extern CUresult (*cuDeviceGetCount)(int *count);
extern CUresult (*cuDeviceGet)(CUdevice *device, int ordinal);
extern CUresult (*cuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev);
extern CUresult (*cuCtxCreate)(CUcontext *pctx, unsigned int flags, CUdevice dev);
extern CUresult (*cuCtxGetCurrent)(CUcontext *pctx);
extern CUresult (*cuCtxGetDevice)(CUdevice *device);
extern CUresult (*cuCtxGetCacheConfig)(CUfunc_cache *pconfig);
extern CUresult (*cuModuleLoadDataEx)(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
extern CUresult (*cuModuleUnload)(CUmodule hmod);
extern CUresult (*cuModuleGetGlobal)(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
extern CUresult (*cuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);
extern CUresult (*cuFuncGetAttribute)(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
extern CUresult (*cuMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
extern CUresult (*cuMemFree)(CUdeviceptr dptr);
extern CUresult (*cuMemsetD8)(CUdeviceptr dstDevice, unsigned char uc, size_t N);
extern CUresult (*cuMemcpyHtoD)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
extern CUresult (*cuMemcpyDtoH)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
extern CUresult (*cuLaunchKernel)(CUfunction f,
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

bool init_cuda();

#endif
