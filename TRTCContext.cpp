#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <string>
#include <string.h>
#include <stdio.h>
#include "TRTCContext.h"
#include "Timing.h"
#include "md5.h"

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

const char* TRTCContext::s_ptx_cache_path = nullptr;

void TRTCContext::set_ptx_cache(const char* path)
{
	s_ptx_cache_path = path;
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


TRTCContext::TRTCContext()
{
	m_verbose = false;
}

TRTCContext::~TRTCContext()
{

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
		int len = p_nl - p;
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

bool TRTCContext::_src_to_ptx(const char* src, std::vector<char>& ptx)
{
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

	size_t ptxSize;
	nvrtcGetPTXSize(prog, &ptxSize);
	ptx.resize(ptxSize);
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
		_src_to_ptx(saxpy.data(), ptx);
		CUmodule module;
		cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0);
		CUdeviceptr dptr;
		cuModuleGetGlobal(&dptr, &size, module, "_test");

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

TRTCContext::KernelTemplate::KernelTemplate(const std::vector<ParamDesc>& params, const char* body, size_t num_reserved_params)
{
	m_params = params;
	m_body = body;
	m_num_reserved_params = num_reserved_params;
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
