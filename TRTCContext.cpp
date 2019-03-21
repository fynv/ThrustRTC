#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <string>
#include <string.h>
#include <stdio.h>
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

struct TRTCContext::Kernel
{
	size_t num_params;
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

bool TRTCContext::_src_to_ptx(const char* src, std::vector<char>& ptx, size_t& ptx_size) const
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

TRTCContext::KernelTemplate::KernelTemplate(const std::vector<const char*> template_params, const std::vector<ParamDesc>& params, const char* body)
{
	m_template_params.resize(template_params.size());
	for (size_t i = 0; i < m_template_params.size(); i++)
		m_template_params[i] = template_params[i];

	m_type_params.resize(params.size());
	for (size_t i = 0; i < m_type_params.size(); i++)
		m_type_params[i] = params[i].type;

	m_code_buf += "extern \"C\" __global__\n";
	m_code_buf += "void saxpy(";

	if (params.size() > 0)
	{
		m_code_buf += params[0].type;
		m_code_buf += " ";
		m_code_buf += params[0].name;
	}

	for (size_t i = 1; i < params.size(); i++)
	{
		m_code_buf += ", ";
		m_code_buf += params[i].type;
		m_code_buf += " ";
		m_code_buf += params[i].name;
	}
	m_code_buf += ")\n{\n";
	m_code_buf += body;
	m_code_buf += "\n}\n";
}

size_t TRTCContext::KernelTemplate::deduce_template_args(DeviceViewable** args, std::vector<std::string>& template_args) const
{
	size_t count = 0;
	size_t total = m_template_params.size();
	template_args.resize(total);

	for (size_t i = 0; i < m_type_params.size(); i++)
	{
		std::string type_param = m_type_params[i];
		std::string type_arg = args[i]->name_view_cls();

		const char* p_type_param = type_param.c_str();
		const char* p_type_arg = type_arg.c_str();

		while (*p_type_param != 0 && *p_type_arg != 0)
		{
			while (*p_type_param == ' ' || *p_type_param == '\t') p_type_param++;
			while (*p_type_arg == ' ' || *p_type_arg == '\t') p_type_arg++;
			if (*p_type_param == 0 || *p_type_arg == 0) break;

			if (*p_type_param != *p_type_arg)
			{
				std::string templ_param;
				std::string templ_arg;
				int ab = 0;
				while (*p_type_param == '_' ||
					(*p_type_param >= 'a' && *p_type_param <= 'z') ||
					(*p_type_param >= 'A' && *p_type_param <= 'Z') ||
					(*p_type_param >= '0' && *p_type_param <= '9'))
					templ_param += *(p_type_param++);
				
				while (*p_type_param == ' ' || *p_type_param == '\t') p_type_param++;
				char end_marker = *p_type_param;

				const char* p_type_arg_end = p_type_arg;
				while (*p_type_arg_end != end_marker) p_type_arg_end++;
				while (*(p_type_arg_end - 1) == ' ' || *(p_type_arg_end - 1) == '\t') p_type_arg_end--;
				while (p_type_arg<p_type_arg_end) templ_arg += *(p_type_arg++);

				size_t j = 0;
				for (; j < total; j++)
				{
					if (templ_param == m_template_params[j])
					{
						if (template_args[j] == "")
						{
							template_args[j] = templ_arg;
							count++;
						}
						else if (template_args[j] != templ_arg)
						{
							printf("Conflict during template-arg deduction of %s, assigned %s before assigning %s.\n", templ_param.c_str(), template_args[j].c_str(), templ_arg.c_str());
							return count;
						}						
						break;
					}
				}
				if (j == total)
				{
					printf("Parameter/argument type mismatch: %s vs. %s\n", type_param.c_str(), type_arg.c_str());
					return count;
				}
			}
			else
			{
				p_type_param++;
				p_type_arg++;
			}
		}
	}
	return count;
}


TRTCContext::Kernel* TRTCContext::KernelTemplate::instantiate(const TRTCContext& ctx, const std::vector<std::string>& template_args) const
{
	std::string saxpy;
	for (size_t i = 0; i < ctx.m_preprocesors.size(); i++)
	{
		saxpy += ctx.m_preprocesors[i] + "\n";
	}
	saxpy += "\n";

	for (size_t i = 0; i < m_template_params.size(); i++)
		saxpy += std::string("#define ") + m_template_params[i] + " " + template_args[i] + "\n";
	
	saxpy += "\n";
	saxpy += m_code_buf;

	if (ctx.m_verbose)
		print_code(saxpy.c_str());

	std::vector<char> ptx;

	{
#ifdef TIMING
		double t1 = GetTime();
#endif
		int compute_cap = s_get_compute_capability();
		char md5[33];

		/// Try finding an existing ptx in cache
		if (s_ptx_cache_path != nullptr)
		{
			s_get_md5(saxpy.c_str(), md5);
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
			if (!ctx._src_to_ptx(saxpy.c_str(), ptx, ptx_size)) return nullptr;
			
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

	//puts(ptx.data());

	Kernel* kernel = new TRTCContext::Kernel;
	kernel->num_params = m_type_params.size();

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

	for (size_t i = 0; i < ctx.m_constants.size(); i++)
	{
		CUdeviceptr dptr;
		size_t size;
		cuModuleGetGlobal(&dptr, &size, kernel->module, ctx.m_constants[i].first.c_str());
		if (size > ctx.m_constants[i].second.size()) size = ctx.m_constants[i].second.size();
		cuMemcpyHtoD(dptr, ctx.m_constants[i].second.data(), size);
	}

	return kernel;	
}

TRTCContext::Kernel* TRTCContext::create_kernel(const std::vector<ParamDesc>& params, const char* body) const
{
	KernelTemplate templ({}, params, body);
	return templ.instantiate(*this, {});
}

size_t TRTCContext::get_num_of_params(const Kernel* kernel)
{
	return kernel->num_params;
}

void TRTCContext::destroy_kernel(Kernel* kernel)
{
	if (kernel)
	{
		cuModuleUnload(kernel->module);
		delete kernel;
	}
}

void TRTCContext::launch_kernel(const Kernel* kernel, dim_type gridDim, dim_type blockDim, DeviceViewable** args, unsigned sharedMemBytes)
{
	if (!kernel) return;

	size_t num_params = kernel->num_params;

	std::vector<ViewBuf> argbufs(num_params);
	std::vector<void*> converted_args(num_params);

	for (size_t i = 0; i < num_params; i++)
	{
		argbufs[i] = args[i]->view();
		converted_args[i] = argbufs[i].data();
	}
	cuLaunchKernel(kernel->func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMemBytes, 0, converted_args.data(), 0);
}

void TRTCContext::launch_once(dim_type gridDim, dim_type blockDim, const std::vector<AssignedParam>& arg_map, const char* code_body, unsigned sharedMemBytes) const
{
	size_t num_params = arg_map.size();
	std::vector<ParamDesc> params(num_params);
	std::vector<std::string> param_types(num_params);
	std::vector<DeviceViewable*> args(num_params);
	for (size_t i = 0; i < num_params; i++)
	{
		param_types[i] = arg_map[i].arg->name_view_cls();
		params[i] = { param_types[i].c_str(), arg_map[i].param_name };
		args[i] = arg_map[i].arg;
	}
	Kernel* ker = create_kernel(params, code_body);
	launch_kernel(ker, gridDim, blockDim, args.data(), sharedMemBytes);
	destroy_kernel(ker);
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