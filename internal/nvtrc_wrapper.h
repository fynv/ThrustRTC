#ifndef _nvrtc_wrapper_h
#define _nvrtc_wrapper_h

#include <cstddef>

typedef struct _nvrtcProgram *nvrtcProgram;

typedef enum {
	NVRTC_SUCCESS = 0,
	NVRTC_ERROR_OUT_OF_MEMORY = 1,
	NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
	NVRTC_ERROR_INVALID_INPUT = 3,
	NVRTC_ERROR_INVALID_PROGRAM = 4,
	NVRTC_ERROR_INVALID_OPTION = 5,
	NVRTC_ERROR_COMPILATION = 6,
	NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
	NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
	NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
	NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
	NVRTC_ERROR_INTERNAL_ERROR = 11
} nvrtcResult;

extern nvrtcResult (*nvrtcVersion)(int *major, int *minor);
extern nvrtcResult (*nvrtcGetNumSupportedArchs)(int* numArchs);
extern nvrtcResult (*nvrtcGetSupportedArchs)(int* supportedArchs);

extern nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram *prog,
	const char *src,
	const char *name,
	int numHeaders,
	const char * const *headers,
	const char * const *includeNames);

extern nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram prog,
	int numOptions, const char * const *options);

extern nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram prog, size_t *logSizeRet);

extern nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram prog, char *log);

extern nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram prog, size_t *ptxSizeRet);

extern nvrtcResult (*nvrtcGetPTX)(nvrtcProgram prog, char *ptx);

extern nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram *prog);

bool init_nvrtc(const char* path_lib = nullptr);



#endif
