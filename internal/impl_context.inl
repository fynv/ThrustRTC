
const char* TRTCContext::s_libnvrtc_path = nullptr;

void TRTCContext::set_libnvrtc_path(const char* path)
{
	static std::string _libnvrtc_path = path;
	s_libnvrtc_path = _libnvrtc_path.c_str();
}

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

static char s_name_db[] = "__ptx_cache__.db";

static bool s_cuda_init(int& cap)
{
	if (!init_cuda())
	{
		printf("Cannot find CUDA driver. \n");
		return false;
	}

	if (!CheckCUresult(cuInit(0), "cuInit()")) return false;

	CUcontext cuContext;
	if (!CheckCUresult(cuCtxGetCurrent(&cuContext), "cuCtxGetCurrent()")) return false;

	if (cuContext == nullptr)
	{
		int max_gflops = 0;
		int max_gflops_device = 0;

		int device_count;
		if (!CheckCUresult(cuDeviceGetCount(&device_count), "cuDeviceGetCount()")) return false;

		if (device_count < 1)
		{
			printf("Cannot find a CUDA capable device. \n");
			return false;
		}
		for (int current_device = 0; current_device < device_count; current_device++)
		{
			CUdevice cuDevice;
			if (!CheckCUresult(cuDeviceGet(&cuDevice, current_device), "cuDeviceGet()")) return false;
			int multiProcessorCount;
			if (!CheckCUresult(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice), "cuDeviceGetAttribute()")) return false;
			int	clockRate;
			if (!CheckCUresult(cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cuDevice), "cuDeviceGetAttribute()")) return false;
			int gflops = multiProcessorCount * clockRate;
			int major, minor;
			if (!CheckCUresult(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice), "cuDeviceGetAttribute()")) return false;
			if (!CheckCUresult(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice), "cuDeviceGetAttribute()")) return false;
			if (major != -1 && minor != -1)
			{
				if (gflops > max_gflops)
				{
					max_gflops = gflops;
					max_gflops_device = current_device;
					cap = major*10 + minor;
				}
			}
		}
		CUdevice cuDevice;
		if (!CheckCUresult(cuDeviceGet(&cuDevice, max_gflops_device), "cuDeviceGet()")) return false;
		if (!CheckCUresult(cuDevicePrimaryCtxRetain(&cuContext, cuDevice), "cuDevicePrimaryCtxRetain()")) return false;
		if (!CheckCUresult(cuCtxSetCurrent(cuContext), "cuCtxSetCurrent()")) return false;
	}
	else
	{
		CUdevice cuDevice;
		if (!CheckCUresult(cuCtxGetDevice(&cuDevice), "cuCtxGetDevice()")) return false;
		if (!CheckCUresult(cuDeviceGetAttribute(&cap, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice), "cuDeviceGetAttribute()")) return false;
	}

	return true;
}

static int s_get_compute_capability(bool istrying=false)
{
	static thread_local int cap = -1;
	if (cap == -1)
	{
		if (!s_cuda_init(cap))
		{
			printf("CUDA initialization failed. \n");
			if (istrying) return -1;
			else exit(0);
		}		
	}
	return cap;
}

bool TRTCContext::try_init()
{
	return s_get_compute_capability(true) != -1;
}

TRTCContext& TRTCContext::get_context()
{
	static TRTCContext s_ctx;
	return s_ctx;
}

static inline unsigned long long s_get_hash(const char* source_code)
{
	uint64_t len = (uint64_t)strlen(source_code);
	return (unsigned long long)crc64(0, (unsigned char*)source_code, len);
}

struct TRTCContext::Kernel
{
	CUmodule module;
	CUfunction func;
	unsigned sharedMemBytes_cached = -1;
	int sizeBlock = -1;
	int numBlocks = -1;
	std::mutex mutex;
};


TRTCContext::TRTCContext()
{
	int v = s_get_compute_capability();

	m_name_header_of_structs = "header_of_structs.h";
	this->add_built_in_header(m_name_header_of_structs.c_str(), m_header_of_structs.c_str());

	m_verbose = false;
	m_kernel_debug = false;
	for (int i = 0; i < s_num_headers; i++)
		this->add_built_in_header(s_name_headers[i], s_content_headers[i]);

	for (int i = 0; i < s_num_headers_global; i++)
		this->add_built_in_header(s_name_headers_global[i], s_content_headers_global[i]);

	this->add_code_block("#define DEVICE_ONLY\n");
	this->add_inlcude_filename("cstdint");
	this->add_inlcude_filename("cfloat");
	this->add_inlcude_filename("cuComplex.h");
	this->add_inlcude_filename("built_in.h");
}

TRTCContext::~TRTCContext()
{
	for (size_t i = 0; i < m_kernel_cache.size(); i++)
	{
		Kernel* kernel = m_kernel_cache[i];
		cuModuleUnload(kernel->module);
		delete kernel;
	}
}

void TRTCContext::set_verbose(bool verbose)
{
	m_verbose = verbose;
}

void TRTCContext::set_kernel_debug(bool kernel_debug)
{
	m_kernel_debug = kernel_debug;
}

static void print_code(const char* name, const char* fullCode)
{
	printf("%s:\n", name);
	const char* p = fullCode;
	int line_num = 1;
	while (true)
	{
		const char* p_nl = strchr(p, '\n');
		if (!p_nl)
			p_nl = p + strlen(p);

		char line[1024];
		int len = (int)(p_nl - p);
		if (len > 1023) len = 1023;
		memcpy(line, p, len);
		line[len] = 0;
		printf("%d\t%s\n", line_num, line);
		if (!*p_nl) break;
		p = p_nl + 1;
		line_num++;
	}
	puts("");
}

void TRTCContext::_get_nvrtc_archs(int& min_arch, int& max_arch)
{
	if (nvrtcGetNumSupportedArchs == nullptr || nvrtcGetSupportedArchs == nullptr)
	{
		int nvrtc_major, nvrtc_minor;
		nvrtcVersion(&nvrtc_major, &nvrtc_minor);

		if (nvrtc_major < 9)
		{
			min_arch = 20;
			max_arch = 53;
		}
		else if (nvrtc_major < 10)
		{
			min_arch = 30;
			max_arch = 72;
		}
		else if (nvrtc_major < 11 && nvrtc_minor < 1)
		{
			min_arch = 30;
			max_arch = 72;
		}
		else if (nvrtc_major < 11)
		{
			min_arch = 30;
			max_arch = 75;
		}
		else if (nvrtc_major < 12 && nvrtc_minor < 1)
		{
			min_arch = 30;
			max_arch = 75;
		}
		else
		{
			min_arch = 35;
			max_arch = 80;
		}
	}
	else
	{
		int numArchs;
		nvrtcGetNumSupportedArchs(&numArchs);
		std::vector<int> archs(numArchs);
		nvrtcGetSupportedArchs(archs.data());
		min_arch = archs[0];
		max_arch = archs[archs.size() - 1];
	}
}

int TRTCContext::get_ptx_arch()
{
	static thread_local int ptx_arch = -1;
	if (ptx_arch == -1)
	{
		if (!init_nvrtc(s_libnvrtc_path))
		{
			printf("Loading libnvrtc failed. \n");
			return -1;
		}

		int compute_cap = s_get_compute_capability();	

		int min_arch, max_arch;
		_get_nvrtc_archs(min_arch, max_arch);

		if (compute_cap <= max_arch)
		{
			if (compute_cap < min_arch)
			{
				printf("NVRTC version too high for GPU (compute_%d).\n", compute_cap);
				return ptx_arch;
			}
			ptx_arch = compute_cap;
		}
		else
		{
			ptx_arch = (max_arch / 10) * 10;
		}		
	}
	return ptx_arch;
}

bool TRTCContext::_src_to_ptx(const char* src, std::vector<char>& ptx, size_t& ptx_size)
{
	if (!init_nvrtc(s_libnvrtc_path))
	{
		printf("Loading libnvrtc failed. \n");
		return false;
	}

	int ptx_arch = get_ptx_arch();
	if (ptx_arch == -1)
	{
		printf("CUDA arch negotiation failed.\n");
		return false;
	}

	int num_headers = (int)m_name_built_in_headers.size();
	std::vector<const char*> p_name_built_in_headers(num_headers);
	std::vector<const char*> p_content_built_in_headers(num_headers);
	for (int i = 0; i < num_headers; i++)
	{
		p_name_built_in_headers[i] = m_name_built_in_headers[i].c_str();
		p_content_built_in_headers[i] = m_content_built_in_headers[i].c_str();
	}

	nvrtcProgram prog;
	nvrtcCreateProgram(&prog, src, "saxpy.cu", num_headers, p_content_built_in_headers.data(), p_name_built_in_headers.data());

	std::vector<std::string> opt_bufs;
	char opt[1024];
	sprintf(opt, "--gpu-architecture=compute_%d", ptx_arch);
	opt_bufs.push_back(opt);

	opt_bufs.push_back("--std=c++14");

	for (size_t i = 0; i < m_include_dirs.size(); i++)
	{
		sprintf(opt, "-I=%s", m_include_dirs[i].c_str());
		opt_bufs.push_back(opt);
	}

	std::vector<const char*> opts(opt_bufs.size());
	for (size_t i = 0; i < opt_bufs.size(); i++)
		opts[i] = opt_bufs[i].c_str();

	nvrtcResult result = NVRTC_SUCCESS;

	result = nvrtcCompileProgram(prog,     // prog
		(int)opts.size(),        // numOptions
		opts.data());    // options

	size_t logSize;
	nvrtcGetProgramLogSize(prog, &logSize);

	if (result != NVRTC_SUCCESS)
	{
		if (!m_verbose)
		{
			{
				std::shared_lock<std::shared_mutex> lock(m_mutex_structs);
				print_code(m_name_header_of_structs.c_str(), m_header_of_structs.c_str());
			}		
			print_code("saxpy.cu", src);
		}

		std::vector<char> log(logSize);
		nvrtcGetProgramLog(prog, log.data());
		puts("Errors:");
		puts(log.data());
		return false;
	}

	nvrtcGetPTXSize(prog, &ptx_size);
	ptx.resize(ptx_size);
	nvrtcGetPTX(prog, ptx.data());
	nvrtcDestroyProgram(&prog);

	return true;
}


size_t TRTCContext::size_of(const char* cls)
{
	std::unique_lock<std::mutex> lock(m_mutex_sizes);
	// try to find in the context cache first
	decltype(m_size_of_types)::iterator it = m_size_of_types.find(cls);
	if (it != m_size_of_types.end())
	{
		size_t size = it->second;
		return size;
	}

	// reflect from device code
	std::string saxpy;
	for (size_t i = 0; i < m_code_blocks.size(); i++)
		saxpy += m_code_blocks[i];
	saxpy += std::string("#include \"") + m_name_header_of_structs + "\"\n";
	saxpy += std::string("__device__ ") + cls + " _test;\n";

	if (m_verbose)
	{
		{
			std::shared_lock<std::shared_mutex> lock(m_mutex_structs);
			print_code(m_name_header_of_structs.c_str(), m_header_of_structs.c_str());
		}		
		print_code("saxpy.cu", saxpy.c_str());
	}

	int compute_cap = s_get_compute_capability();
	unsigned long long hash;

	size_t size = (size_t)(-1);

	/// Try finding an existing ptx in disk cache
	{
		hash = s_get_hash(saxpy.c_str());
		char key[64];
		sprintf(key, "%016llx_%d", hash, compute_cap);
		unqlite *pDb;
		if (UNQLITE_OK == unqlite_open(&pDb, s_name_db, UNQLITE_OPEN_CREATE))
		{
			unqlite_int64 nBytes = sizeof(size_t);
			unqlite_kv_fetch(pDb, key, -1, &size, &nBytes);
			unqlite_close(pDb);
		}
	}

	if (size == (size_t)(-1))
	{
		std::vector<char> ptx;
		size_t ptx_size;
		if (!_src_to_ptx(saxpy.data(), ptx, ptx_size)) return (size_t)(-1);
		CUmodule module;
		if (!CheckCUresult(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0), "cuModuleLoadDataEx()")) return (size_t)(-1);
		CUdeviceptr dptr;
		if (!CheckCUresult(cuModuleGetGlobal(&dptr, &size, module, "_test"), "cuModuleGetGlobal()")) return (size_t)(-1);
		if (!CheckCUresult(cuModuleUnload(module), "cuModuleUnload()")) return (size_t)(-1);

		{
			char key[64];
			sprintf(key, "%016llx_%d", hash, compute_cap);
			unqlite *pDb;
			if (UNQLITE_OK == unqlite_open(&pDb, s_name_db, UNQLITE_OPEN_CREATE))
			{
				unqlite_kv_store(pDb, key, -1, &size, sizeof(size_t));
				unqlite_close(pDb);
			}
		}
	}

	// cache the result
	m_size_of_types[cls] = size;

	return size;
}

bool TRTCContext::query_struct(const char* name_struct, const std::vector<const char*>& name_members, size_t* offsets)
{
	// handle simple cases
	if (name_members.size() == 0)
	{
		offsets[0] = 1;
		return true;
	}
	else if (name_members.size() == 1)
	{
		offsets[0] = 0;
		offsets[1] = size_of(name_struct);
		return offsets[1] != (size_t)(-1);
	}

	std::unique_lock<std::mutex> lock(m_mutex_offsets);

	// try to find in the context cache first
	decltype(m_offsets_of_structs)::iterator it = m_offsets_of_structs.find(name_struct);
	if (it != m_offsets_of_structs.end())
	{
		memcpy(offsets, it->second.data(), sizeof(size_t)*it->second.size());
		return true;
	}

	// reflect from device code
	std::vector<size_t> res(name_members.size() + 1);

	std::string saxpy;
	for (size_t i = 0; i < m_code_blocks.size(); i++)
		saxpy += m_code_blocks[i];
	saxpy += std::string("#include \"") + m_name_header_of_structs + "\"\n";
	saxpy += std::string("__device__ ") + name_struct + " _test;\n";

	char line[1024];
	sprintf(line, "__device__ size_t _res[%u] = {", (unsigned)name_members.size() + 1);
	saxpy += line;

	for (size_t i = 0; i < name_members.size(); i++)
	{
		sprintf(line, "(char*)&_test.%s - (char*)&_test, ", name_members[i]);
		saxpy += line;
	}
	saxpy += "sizeof(_test)};\n";

	if (m_verbose)
	{
		{
			std::shared_lock<std::shared_mutex> lock(m_mutex_structs);
			print_code(m_name_header_of_structs.c_str(), m_header_of_structs.c_str());
		}		
		print_code("saxpy.cu", saxpy.c_str());
	}

	int compute_cap = s_get_compute_capability();
	unsigned long long hash;

	bool loaded = false;

	/// Try finding an existing ptx in disk cache
	{
		hash = s_get_hash(saxpy.c_str());
		char key[64];
		sprintf(key, "%016llx_%d", hash, compute_cap);
		unqlite *pDb;
		if (UNQLITE_OK == unqlite_open(&pDb, s_name_db, UNQLITE_OPEN_CREATE))
		{
			unqlite_int64 nBytes = res.size() * sizeof(size_t);
			if (UNQLITE_OK == unqlite_kv_fetch(pDb, key, -1, res.data(), &nBytes))
				loaded = true;
			unqlite_close(pDb);
		}
	}

	if (!loaded)
	{
		std::vector<char> ptx;
		size_t ptx_size;
		if (!_src_to_ptx(saxpy.data(), ptx, ptx_size)) return false;

		CUmodule module;
		if (!CheckCUresult(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0), "cuModuleLoadDataEx()")) return false;
		size_t size_res;
		CUdeviceptr dptr_res;
		if (!CheckCUresult(cuModuleGetGlobal(&dptr_res, &size_res, module, "_res"), "cuModuleGetGlobal()")) return false;
		if (!CheckCUresult(cuMemcpyDtoH(res.data(), dptr_res, size_res), "cuMemcpyDtoH()")) return false;
		if (!CheckCUresult(cuModuleUnload(module), "cuModuleUnload()")) return false;

		{
			char key[64];
			sprintf(key, "%016llx_%d", hash, compute_cap);
			unqlite *pDb;
			if (UNQLITE_OK == unqlite_open(&pDb, s_name_db, UNQLITE_OPEN_CREATE))
			{
				unqlite_kv_store(pDb, key, -1, res.data(), res.size() * sizeof(size_t));
				unqlite_close(pDb);
			}
		}
	}

	// cache the result
	m_offsets_of_structs[name_struct] = res;

	memcpy(offsets, res.data(), sizeof(size_t)*res.size());
	return true;
}

KernelId_t TRTCContext::_build_kernel(const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body, std::string* p_saxpy)
{
	std::string saxpy;
	for (size_t i = 0; i < m_code_blocks.size(); i++)
	{
		saxpy += m_code_blocks[i];
	}
	saxpy += std::string("#include \"") + m_name_header_of_structs + "\"\n";

	saxpy += "\n";
	saxpy += "extern \"C\" __global__\n";
	saxpy += "void saxpy(";

	size_t num_params = arg_map.size();

	if (num_params > 0)
	{
		saxpy += arg_map[0].obj->name_view_cls();
		saxpy += " ";
		saxpy += arg_map[0].obj_name;
	}

	for (size_t i = 1; i < num_params; i++)
	{
		saxpy += ", ";
		saxpy += arg_map[i].obj->name_view_cls();
		saxpy += " ";
		saxpy += arg_map[i].obj_name;
	}

	saxpy += ")\n{\n";
	saxpy += code_body;
	saxpy += "\n}\n";

	if (m_verbose)
	{
		{
			std::shared_lock<std::shared_mutex> lock(m_mutex_structs);
			print_code(m_name_header_of_structs.c_str(), m_header_of_structs.c_str());
		}
		print_code("saxpy.cu", saxpy.c_str());
	}

	if (p_saxpy != nullptr)
	{
		*p_saxpy = saxpy;
	}

	unsigned long long hash = s_get_hash(saxpy.c_str());
	KernelId_t kid = (KernelId_t)(-1);

	std::unique_lock<std::shared_mutex> lock(m_mutex_kernels);

	decltype(m_kernel_id_map)::iterator it = m_kernel_id_map.find(hash);
	if (it != m_kernel_id_map.end())
	{
		kid = it->second;
		return kid;
	}

	std::vector<char> ptx;
	{
		int compute_cap = s_get_compute_capability();

		/// Try finding an existing ptx in cache
		{
			char key[64];
			sprintf(key, "%016llx_%d", hash, compute_cap);
			unqlite *pDb;
			if (UNQLITE_OK == unqlite_open(&pDb, s_name_db, UNQLITE_OPEN_CREATE))
			{
				unqlite_int64 nBytes;
				if (UNQLITE_OK == unqlite_kv_fetch(pDb, key, -1, NULL, &nBytes))
				{
					size_t ptx_size = nBytes + 1;
					ptx.resize(ptx_size);
					unqlite_kv_fetch(pDb, key, -1, ptx.data(), &nBytes);
					ptx[ptx_size - 1] = 0;
				}
				unqlite_close(pDb);
			}
		}
		if (ptx.size() < 1)
		{
			size_t ptx_size;
			if (!_src_to_ptx(saxpy.c_str(), ptx, ptx_size)) return kid;

			{
				char key[64];
				sprintf(key, "%016llx_%d", hash, compute_cap);
				unqlite *pDb;
				if (UNQLITE_OK == unqlite_open(&pDb, s_name_db, UNQLITE_OPEN_CREATE))
				{
					unqlite_kv_store(pDb, key, -1, ptx.data(), ptx_size - 1);
					unqlite_close(pDb);
				}
			}
		}
	}

	Kernel* kernel = new Kernel;

	{
		if (!CheckCUresult(cuModuleLoadDataEx(&kernel->module, ptx.data(), 0, 0, 0), "cuModuleLoadDataEx()")) return kid;
		if (!CheckCUresult(cuModuleGetFunction(&kernel->func, kernel->module, "saxpy"), "cuModuleGetFunction()")) return kid;
	}
	for (size_t i = 0; i < m_constants.size(); i++)
	{
		CUdeviceptr dptr;
		size_t size;
		if (!CheckCUresult(cuModuleGetGlobal(&dptr, &size, kernel->module, m_constants[i].first.c_str()), "cuModuleGetGlobal()")) return kid;
		if (size > m_constants[i].second.size()) size = m_constants[i].second.size();
		if (!CheckCUresult(cuMemcpyHtoD(dptr, m_constants[i].second.data(), size), "cuMemcpyHtoD()")) return kid;
	}

	m_kernel_cache.push_back(kernel);
	kid = (unsigned)m_kernel_cache.size() - 1;
	m_kernel_id_map[hash] = kid;

	return kid;
}

int TRTCContext::_launch_calc(KernelId_t kid, unsigned sharedMemBytes)
{
	Kernel *kernel;
	{
		std::shared_lock<std::shared_mutex> lock(m_mutex_kernels);
		kernel = m_kernel_cache[kid];
	}

	int size;
	{
		std::unique_lock<std::mutex> lock(kernel->mutex);
		if (sharedMemBytes == kernel->sharedMemBytes_cached)
		{
			size = kernel->sizeBlock;
			return size;
		}
	
		if (!launch_calc(kernel->func, sharedMemBytes, kernel->sizeBlock)) return -1;
		kernel->sharedMemBytes_cached = sharedMemBytes;
		size = kernel->sizeBlock;
	}
	return size;
}

int TRTCContext::_persist_calc(KernelId_t kid, int sizeBlock, unsigned sharedMemBytes)
{
	Kernel *kernel;
	{
		std::shared_lock<std::shared_mutex> lock(m_mutex_kernels);
		kernel = m_kernel_cache[kid];
	}

	int num;
	{
		std::unique_lock<std::mutex> lock(kernel->mutex);
		if (sharedMemBytes == kernel->sharedMemBytes_cached && sizeBlock == kernel->sizeBlock)
		{
			num = kernel->numBlocks;
			return num;
		}
	
		persist_calc(kernel->func, sharedMemBytes, sizeBlock, kernel->numBlocks);
		kernel->sharedMemBytes_cached = sharedMemBytes;
		kernel->sizeBlock = sizeBlock;
		num = kernel->numBlocks;
	}
	return num;
}

bool TRTCContext::_launch_kernel(KernelId_t kid, dim_type gridDim, dim_type blockDim, const std::vector<CapturedDeviceViewable>& arg_map, unsigned sharedMemBytes)
{
	Kernel *kernel;
	{
		std::shared_lock<std::shared_mutex> lock(m_mutex_kernels);
		kernel = m_kernel_cache[kid];
	}
	size_t num_params = arg_map.size();
	std::vector<ViewBuf> argbufs(num_params);
	std::vector<void*> converted_args(num_params);

	for (size_t i = 0; i < num_params; i++)
	{
		argbufs[i] = arg_map[i].obj->view();
		converted_args[i] = argbufs[i].data();
	}
	CUresult res = cuLaunchKernel(kernel->func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMemBytes, 0, converted_args.data(), 0);
	if (!CheckCUresult(res, "cuLaunchKernel()")) return false;
	return true;
}

bool TRTCContext::calc_optimal_block_size(const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body, int& sizeBlock, unsigned sharedMemBytes)
{
	KernelId_t kid = _build_kernel(arg_map, code_body);
	if (kid == (KernelId_t)(-1)) return false;
	int size_block = _launch_calc(kid, sharedMemBytes);
	if (size_block < 0) return false;	
	sizeBlock = size_block;
	return true;
}

bool TRTCContext::calc_number_blocks(const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body, int sizeBlock, int& numBlocks, unsigned sharedMemBytes)
{
	KernelId_t kid = _build_kernel(arg_map, code_body);
	if (kid == (KernelId_t)(-1)) return false;
	numBlocks = _persist_calc(kid, sizeBlock, sharedMemBytes);
	return true;
}

bool TRTCContext::launch_kernel(dim_type gridDim, dim_type blockDim, const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body, unsigned sharedMemBytes)
{
	KernelId_t kid;
	std::string saxpy;
	if (m_kernel_debug)
	{
		kid = _build_kernel(arg_map, code_body, &saxpy);
	}
	else
	{
		kid = _build_kernel(arg_map, code_body);
	}
	if (kid == (KernelId_t)(-1))
	{
		printf("Failed to build kernel\n");
		return false;
	}

	if (!_launch_kernel(kid, gridDim, blockDim, arg_map, sharedMemBytes)) return false;
	if (m_kernel_debug)
	{
		if (!CheckCUresult(cuCtxSynchronize(), "cuCtxSynchronize()"))
		{
			if (!m_verbose)
			{
				{
					std::shared_lock<std::shared_mutex> lock(m_mutex_structs);
					print_code(m_name_header_of_structs.c_str(), m_header_of_structs.c_str());
				}
				print_code("saxpy.cu", saxpy.c_str());
			}
			return false;
		}
	}
	return true;
}

bool TRTCContext::launch_for(size_t begin, size_t end, const std::vector<CapturedDeviceViewable>& _arg_map, const char* name_iter, const char* _body)
{
	DVSizeT dvbegin(begin), dvend(end);
	Functor func(_arg_map, { name_iter }, _body);
	std::vector<CapturedDeviceViewable> arg_map = { {"begin", &dvbegin}, {"end", &dvend}, {"func", &func} };
	std::string code_body =
		"    size_t tid =  threadIdx.x + blockIdx.x*blockDim.x + begin;\n"
		"    if(tid>=end) return;\n"
		"    func(tid);\n";

	KernelId_t kid;
	std::string saxpy;
	if (m_kernel_debug)
	{
		kid = _build_kernel(arg_map, code_body.c_str(), &saxpy);
	}
	else
	{
		kid = _build_kernel(arg_map, code_body.c_str());
	}

	if (kid == (KernelId_t)(-1))
	{
		printf("Failed to build kernel\n");
		return false;
	}
	int size_block = _launch_calc(kid, 0);
	if (size_block < 0) return false;
	unsigned sizeBlock = (unsigned)size_block;
	unsigned numBlocks = (unsigned)((end - begin + sizeBlock - 1) / sizeBlock);

	if (!_launch_kernel(kid, { numBlocks, 1, 1 }, { sizeBlock, 1, 1 }, arg_map, 0)) return false;
	if (m_kernel_debug)
	{
		if (!CheckCUresult(cuCtxSynchronize(), "cuCtxSynchronize()"))
		{
			if (!m_verbose)
			{
				{
					std::shared_lock<std::shared_mutex> lock(m_mutex_structs);
					print_code(m_name_header_of_structs.c_str(), m_header_of_structs.c_str());
				}
				print_code("saxpy.cu", saxpy.c_str());
			}
			return false;
		}
	}
	return true;
}

bool TRTCContext::launch_for_n(size_t n, const std::vector<CapturedDeviceViewable>& _arg_map, const char* name_iter, const char* _body)
{
	DVSizeT dv_n(n);
	Functor func(_arg_map, { name_iter }, _body);
	std::vector<CapturedDeviceViewable> arg_map = { {"n", &dv_n}, {"func", &func} };
	std::string code_body =
		"    size_t tid =  threadIdx.x + blockIdx.x*blockDim.x;\n"
		"    if(tid>=n) return;\n"
		"    func(tid);\n";

	KernelId_t kid;
	std::string saxpy;
	if (m_kernel_debug)
	{
		kid = _build_kernel(arg_map, code_body.c_str(), &saxpy);
	}
	else
	{
		kid = _build_kernel(arg_map, code_body.c_str());
	}

	if (kid == (KernelId_t)(-1))
	{
		printf("Failed to build kernel\n");
		return false;
	}
	int size_block = _launch_calc(kid, 0);
	if (size_block < 0) return false;
	unsigned sizeBlock = (unsigned)size_block;
	unsigned numBlocks = (unsigned)((n + sizeBlock - 1) / sizeBlock);
	if (!_launch_kernel(kid, { numBlocks, 1, 1 }, { sizeBlock, 1, 1 }, arg_map, 0)) return false;
	if (m_kernel_debug)
	{
		if (!CheckCUresult(cuCtxSynchronize(), "cuCtxSynchronize()"))
		{
			if (!m_verbose)
			{
				{
					std::shared_lock<std::shared_mutex> lock(m_mutex_structs);
					print_code(m_name_header_of_structs.c_str(), m_header_of_structs.c_str());
				}
				print_code("saxpy.cu", saxpy.c_str());
			}
			return false;
		}
	}
	return true;
}

void TRTCContext::add_include_dir(const char* path)
{
	m_include_dirs.push_back(path);
}

void TRTCContext::add_built_in_header(const char* name, const char* content)
{
	m_name_built_in_headers.push_back(name);
	m_content_built_in_headers.push_back(content);
}

void TRTCContext::add_code_block(const char* code)
{
	m_code_blocks.push_back(code);
}

void TRTCContext::add_inlcude_filename(const char* fn)
{
	char line[1024];
	sprintf(line, "#include \"%s\"\n", fn);
	add_code_block(line);
}

void TRTCContext::add_constant_object(const char* name, const DeviceViewable& obj)
{
	std::string type = obj.name_view_cls();
	char line[1024];
	sprintf(line, "__constant__ %s %s;\n", type.c_str(), name);
	add_code_block(line);
	m_constants.push_back({ name, obj.view() });
}

std::string TRTCContext::add_struct(const char* struct_body)
{
	unsigned long long hash = s_get_hash(struct_body);
	char name[32];
	sprintf(name, "_S_%016llx", hash);

	std::unique_lock<std::shared_mutex> lock(m_mutex_structs);
	decltype(m_known_structs)::iterator it = m_known_structs.find(hash);
	if (it != m_known_structs.end())
		return name;
	

	std::string struct_def = std::string("struct ") + name + "\n{\n"
		"    typedef " + name + " CurType;\n" +
		struct_body + "};\n";
	
	m_header_of_structs += struct_def;
	m_content_built_in_headers[0] = m_header_of_structs.c_str();
	m_known_structs.insert(hash);

	return name;
}
