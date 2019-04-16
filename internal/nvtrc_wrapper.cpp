#include "nvtrc_wrapper.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <dlfcn.h>
#endif

#include <stdio.h>

static bool s_nvrtc_initialized = false;

nvrtcResult(*nvrtcCreateProgram)(nvrtcProgram *prog,
	const char *src,
	const char *name,
	int numHeaders,
	const char * const *headers,
	const char * const *includeNames);

nvrtcResult(*nvrtcCompileProgram)(nvrtcProgram prog,
	int numOptions, const char * const *options);

nvrtcResult(*nvrtcGetProgramLogSize)(nvrtcProgram prog, size_t *logSizeRet);

nvrtcResult(*nvrtcGetProgramLog)(nvrtcProgram prog, char *log);

nvrtcResult(*nvrtcGetPTXSize)(nvrtcProgram prog, size_t *ptxSizeRet);

nvrtcResult(*nvrtcGetPTX)(nvrtcProgram prog, char *ptx);

nvrtcResult(*nvrtcDestroyProgram)(nvrtcProgram *prog);

bool init_nvrtc(const char* path_lib)
{
	if (s_nvrtc_initialized) return true;
#ifdef _WIN32
	char path_dll[1024];
	if (path_lib == nullptr)
	{
		char cuda_path[1024];
		char cuda_bin_path[1024];
		if (!GetEnvironmentVariable("CUDA_PATH", cuda_path, 1024))
			sprintf(cuda_bin_path, ".");
		else
			sprintf(cuda_bin_path, "%s\\bin", cuda_path);			

		WIN32_FIND_DATAA ffd;
		HANDLE hFind = INVALID_HANDLE_VALUE;

		char search_str[1024];
		sprintf(search_str, "%s\\nvrtc64*", cuda_bin_path);
		hFind = FindFirstFileA(search_str, &ffd);
		if (INVALID_HANDLE_VALUE == hFind) return false;

		sprintf(path_dll, "%s\\%s", cuda_bin_path, ffd.cFileName);
		path_lib = path_dll;
	}

	HINSTANCE hinstLib = LoadLibraryA(path_lib);
	if (hinstLib == NULL)
	{
		printf("nvrtc not found at %s\n", path_lib);
		return false;
	}

	nvrtcCreateProgram = (decltype(nvrtcCreateProgram))GetProcAddress(hinstLib, "nvrtcCreateProgram");
	nvrtcCompileProgram = (decltype(nvrtcCompileProgram))GetProcAddress(hinstLib, "nvrtcCompileProgram");
	nvrtcGetProgramLogSize = (decltype(nvrtcGetProgramLogSize))GetProcAddress(hinstLib, "nvrtcGetProgramLogSize");
	nvrtcGetProgramLog = (decltype(nvrtcGetProgramLog))GetProcAddress(hinstLib, "nvrtcGetProgramLog");
	nvrtcGetPTXSize = (decltype(nvrtcGetPTXSize))GetProcAddress(hinstLib, "nvrtcGetPTXSize");
	nvrtcGetPTX = (decltype(nvrtcGetPTX))GetProcAddress(hinstLib, "nvrtcGetPTX");
	nvrtcDestroyProgram = (decltype(nvrtcDestroyProgram))GetProcAddress(hinstLib, "nvrtcDestroyProgram");
	
#else
	char default_path_so[] = "/usr/local/cuda/lib64/libnvrtc.so";
	if (path_lib == nullptr)
		path_lib = default_path_so;

	void *handle = dlopen(path_lib, RTLD_LAZY);
	if (!handle)
	{
		printf("nvrtc not found at %s\n", path_lib);
		return false;
	}
	
	nvrtcCreateProgram = (decltype(nvrtcCreateProgram))dlsym(handle, "nvrtcCreateProgram");
	nvrtcCompileProgram = (decltype(nvrtcCompileProgram))dlsym(handle, "nvrtcCompileProgram");
	nvrtcGetProgramLogSize = (decltype(nvrtcGetProgramLogSize))dlsym(handle, "nvrtcGetProgramLogSize");
	nvrtcGetProgramLog = (decltype(nvrtcGetProgramLog))dlsym(handle, "nvrtcGetProgramLog");
	nvrtcGetPTXSize = (decltype(nvrtcGetPTXSize))dlsym(handle, "nvrtcGetPTXSize");
	nvrtcGetPTX = (decltype(nvrtcGetPTX))dlsym(handle, "nvrtcGetPTX");
	nvrtcDestroyProgram = (decltype(nvrtcDestroyProgram))dlsym(handle, "nvrtcDestroyProgram");

#endif

	s_nvrtc_initialized = true;
	return true;
}

