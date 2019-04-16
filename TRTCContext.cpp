#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <string.h>
#include <stdio.h>
#include "nvtrc_wrapper.h"
#include "TRTCContext.h"
#include "Timing.h"
#include "md5.h"
#include "cuda_inline_headers.hpp"
#include "cuda_inline_headers_global.hpp"

static int s_get_compute_capability()
{
	static int cap = -1;
	if (cap == -1)
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		cap = devProp.major;
		if (cap < 2 || cap>7) cap = 7;
		cudaFree(0);
	}
	return cap;
}

const char* TRTCContext::s_libnvrtc_path = nullptr;
const char* TRTCContext::s_ptx_cache_path = nullptr;

void TRTCContext::set_libnvrtc_path(const char* path)
{
	static std::string _libnvrtc_path = path;
	s_libnvrtc_path = _libnvrtc_path.c_str();
}

void TRTCContext::set_ptx_cache(const char* path)
{
	static std::string _ptx_cache_path = path;
	s_ptx_cache_path = _ptx_cache_path.c_str();
}

static void s_get_md5(const char* source_code, char md5[33])
{
	MD5Context ctx;
	MD5Init(&ctx);
	unsigned len = (unsigned)strlen(source_code);
	MD5Update(&ctx, (const unsigned char*)source_code, len);
	unsigned char digest[16];
	MD5Final(digest, &ctx);
	sprintf(md5, "%08x%08x%08x%08x", ((unsigned*)digest)[3], ((unsigned*)digest)[2], ((unsigned*)digest)[1], ((unsigned*)digest)[0]);
}

struct TRTCContext::Kernel
{
	CUmodule module;
	CUfunction func;
};


TRTCContext::TRTCContext()
{
	m_verbose = false;
	for (int i = 0; i < s_num_headers; i++)
		this->add_built_in_header(s_name_headers[i], s_content_headers[i]);

	for (int i = 0; i < s_num_headers_global; i++)
		this->add_built_in_header(s_name_headers_global[i], s_content_headers_global[i]);

	this->add_preprocessor("#define DEVICE_ONLY");
	this->add_inlcude_filename("DVVector.h");
	this->add_inlcude_filename("cstdint");
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

static void print_code(const char* fullCode)
{
	puts("saxpy.cu:");
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

bool TRTCContext::_src_to_ptx(const char* src, std::vector<char>& ptx, size_t& ptx_size) const
{
	if (!init_nvrtc(s_libnvrtc_path))
	{
		printf("Loading libnvrtc failed.\n");
		return false;
	}

	int compute_cap = s_get_compute_capability();

	nvrtcProgram prog;
	nvrtcCreateProgram(&prog,         // prog
		src,         // buffer
		"saxpy.cu",    // name
		(int)m_name_built_in_headers.size(),             // numHeaders
		m_content_built_in_headers.data(),          // headers
		m_name_built_in_headers.data());         // includeNames

	std::vector<std::string> opt_bufs;
	char opt[1024];
	sprintf(opt, "--gpu-architecture=compute_%d0", compute_cap);
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
		if (!m_verbose)	print_code(src);

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
	// try to find in the context cache first
	std::unordered_map<std::string, size_t>::iterator it = m_size_of_types.find(cls);
	if (it != m_size_of_types.end()) return it->second;

	// reflect from device code
	std::string saxpy;
	for (size_t i = 0; i < m_preprocesors.size(); i++)
		saxpy += m_preprocesors[i] + "\n";
	saxpy += std::string("__device__ ") + cls + " _test;\n";

	if (m_verbose) print_code(saxpy.c_str());

	int compute_cap = s_get_compute_capability();
	char md5[33];

	size_t size=(size_t)(-1);

	/// Try finding an existing ptx in disk cache
	if (s_ptx_cache_path != nullptr)
	{
		s_get_md5(saxpy.c_str(), md5);
		char fn[2048];
		sprintf(fn, "%s/%s_%d.size", s_ptx_cache_path, md5, compute_cap);
		FILE* fp = fopen(fn, "rb");
		if (fp)
		{
			fread(&size, 1, sizeof(size_t), fp);
			fclose(fp);
		}
	}

	if (size == (size_t)(-1))
	{
		std::vector<char> ptx;
		size_t ptx_size;
		if (!_src_to_ptx(saxpy.data(), ptx, ptx_size)) return 0;
		CUmodule module;
		cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0);
		CUdeviceptr dptr;
		cuModuleGetGlobal(&dptr, &size, module, "_test");
		cuModuleUnload(module);

		if (s_ptx_cache_path != nullptr)
		{
			char fn[2048];
			sprintf(fn, "%s/%s_%d.size", s_ptx_cache_path, md5, compute_cap);
			FILE* fp = fopen(fn, "wb");
			if (fp)
			{
				fwrite(&size, 1, sizeof(size_t), fp);
				fclose(fp);
			}
		}
	}

	// cache the result
	m_size_of_types[cls] = size;

	return size;
}

bool TRTCContext::launch_kernel(dim_type gridDim, dim_type blockDim, const std::vector<AssignedParam>& arg_map, const char* code_body, unsigned sharedMemBytes)
{
	std::string saxpy;
	for (size_t i = 0; i < m_preprocesors.size(); i++)
	{
		saxpy += m_preprocesors[i] + "\n";
	}

	saxpy += "\n";
	saxpy += "extern \"C\" __global__\n";
	saxpy += "void saxpy(";

	size_t num_params = arg_map.size();

	if (num_params > 0)
	{
		saxpy += arg_map[0].arg->name_view_cls();
		saxpy += " ";
		saxpy += arg_map[0].param_name;
	}

	for (size_t i = 1; i < num_params; i++)
	{
		saxpy += ", ";
		saxpy += arg_map[i].arg->name_view_cls();
		saxpy += " ";
		saxpy += arg_map[i].param_name;
	}

	saxpy += ")\n{\n";
	saxpy += code_body;
	saxpy += "\n}\n";

	if (m_verbose)
		print_code(saxpy.c_str());

	char md5[33];
	s_get_md5(saxpy.c_str(), md5);

	KernelId_t kid = (KernelId_t)(-1);
	do
	{
		{
			std::unordered_map<std::string, KernelId_t>::iterator it = m_kernel_id_map.find(md5);
			if (it != m_kernel_id_map.end())
			{
				kid = it->second;
				break;
			}
		}

		std::vector<char> ptx;
		{
#ifdef TIMING
			double t1 = GetTime();
#endif
			int compute_cap = s_get_compute_capability();

			/// Try finding an existing ptx in cache
			if (s_ptx_cache_path != nullptr)
			{
				char fn[2048];
				sprintf(fn, "%s/%s_%d.ptx", s_ptx_cache_path, md5, compute_cap);
				FILE* fp = fopen(fn, "rb");
				if (fp)
				{
					fseek(fp, 0, SEEK_END);
					size_t ptx_size = (size_t)ftell(fp) + 1;
					fseek(fp, 0, SEEK_SET);
					ptx.resize(ptx_size);
					fread(ptx.data(), 1, ptx_size - 1, fp);
					fclose(fp);
					ptx[ptx_size - 1] = 0;
				}
			}
			if (ptx.size() < 1)
			{
				size_t ptx_size;
				if (!_src_to_ptx(saxpy.c_str(), ptx, ptx_size)) break;

				if (s_ptx_cache_path != nullptr)
				{
					char fn[2048];
					sprintf(fn, "%s/%s_%d.ptx", s_ptx_cache_path, md5, compute_cap);
					FILE* fp = fopen(fn, "wb");
					if (fp)
					{
						fwrite(ptx.data(), 1, ptx_size - 1, fp);
						fclose(fp);
					}
				}
			}

#ifdef TIMING
			double t2 = GetTime();
			printf("Compile Time: %f\n", t2 - t1);
#endif
		}

		Kernel* kernel = new Kernel;

		{
#ifdef TIMING
			double t1 = GetTime();
#endif
			cuModuleLoadDataEx(&kernel->module, ptx.data(), 0, 0, 0);
			cuModuleGetFunction(&kernel->func, kernel->module, "saxpy");
#ifdef TIMING
			double t2 = GetTime();
			printf("Load Time: %f\n", t2 - t1);
#endif
		}
		for (size_t i = 0; i < m_constants.size(); i++)
		{
			CUdeviceptr dptr;
			size_t size;
			cuModuleGetGlobal(&dptr, &size, kernel->module, m_constants[i].first.c_str());
			if (size > m_constants[i].second.size()) size = m_constants[i].second.size();
			cuMemcpyHtoD(dptr, m_constants[i].second.data(), size);
		}
		m_kernel_cache.push_back(kernel);
		kid = (unsigned)m_kernel_cache.size() - 1;
		m_kernel_id_map[md5] = kid;
	} while (false);

	if (kid == (KernelId_t)(-1)) return false;

	Kernel *kernel = m_kernel_cache[kid];
	std::vector<ViewBuf> argbufs(num_params);
	std::vector<void*> converted_args(num_params);

	for (size_t i = 0; i < num_params; i++)
	{
		argbufs[i] = arg_map[i].arg->view();
		converted_args[i] = argbufs[i].data();
	}
	CUresult res= cuLaunchKernel(kernel->func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMemBytes, 0, converted_args.data(), 0);

	return res == CUDA_SUCCESS;
}


bool TRTCContext::launch_for(size_t begin, size_t end, const std::vector<TRTCContext::AssignedParam>& _arg_map, const char* name_iter, const char* _body, unsigned sharedMemBytes)
{
	DVSizeT dvbegin(begin), dvend(end);
	std::vector<TRTCContext::AssignedParam> arg_map = _arg_map;
	arg_map.push_back({ "_begin", &dvbegin });
	arg_map.push_back({ "_end", &dvend });

	std::string body = std::string("    size_t ") + name_iter + " = threadIdx.x + blockIdx.x*blockDim.x + _begin;\n"
		"    if (" + name_iter + ">=_end) return; \n" + _body;

	unsigned num_blocks = (unsigned)((end - begin + 127) / 128);
	return launch_kernel({ num_blocks, 1, 1 }, { 128, 1, 1 }, arg_map, body.c_str(), sharedMemBytes);
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

void TRTCContext::add_inlcude_filename(const char* fn)
{
	char line[1024];
	sprintf(line, "#include \"%s\"", fn);
	m_preprocesors.push_back(line);
}

void TRTCContext::add_preprocessor(const char* line)
{
	m_preprocesors.push_back(line);
}

void TRTCContext::add_constant_object(const char* name, const DeviceViewable& obj)
{
	std::string type = obj.name_view_cls();
	char line[1024];
	sprintf(line, "__constant__ %s %s;", type.c_str(), name);
	m_preprocesors.push_back(line);
	m_constants.push_back({ name, obj.view() });
}

TRTC_Kernel::TRTC_Kernel(const std::vector<const char*>& param_names, const char* code_body) :
m_param_names(param_names.size()), m_code_body(code_body)
{
	for (size_t i = 0; i < param_names.size(); i++)
		m_param_names[i] = param_names[i];
}

bool TRTC_Kernel::launch(TRTCContext& ctx, dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes)
{
	std::vector<TRTCContext::AssignedParam> arg_map(m_param_names.size());
	for (size_t i = 0; i < m_param_names.size(); i++)
	{
		arg_map[i].param_name = m_param_names[i].c_str();
		arg_map[i].arg = args[i];
	}
	return ctx.launch_kernel(gridDim, blockDim, arg_map, m_code_body.c_str(), sharedMemBytes);
}

TRTC_For::TRTC_For(const std::vector<const char*>& param_names, const char* name_iter, const char* code_body) :
m_param_names(param_names.size()), m_name_iter(name_iter), m_code_body(code_body)
{
	for (size_t i = 0; i < param_names.size(); i++)
		m_param_names[i] = param_names[i];
}

bool TRTC_For::launch(TRTCContext& ctx, size_t begin, size_t end, const DeviceViewable** args, unsigned sharedMemBytes)
{
	std::vector<TRTCContext::AssignedParam> arg_map(m_param_names.size());
	for (size_t i = 0; i < m_param_names.size(); i++)
	{
		arg_map[i].param_name = m_param_names[i].c_str();
		arg_map[i].arg = args[i];
	}
	return ctx.launch_for(begin, end, arg_map, m_name_iter.c_str(), m_code_body.c_str(), sharedMemBytes);
}