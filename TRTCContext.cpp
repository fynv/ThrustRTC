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

TRTCContext::KernelTemplate::KernelTemplate(const std::vector<const char*>& template_params, const std::vector<ParamDesc>& params, const char* body)
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

static void s_tokenize(const char* in, std::vector<std::string>& out)
{
	const char* pin = in;
	std::string cur_token;
	out.clear();
	while (*pin != 0)
	{
		if (*pin == '_' || (*pin >= 'a' && *pin <= 'z') || (*pin >= 'A' && *pin <= 'Z') || (*pin >= '0' && *pin <= '9'))
			cur_token += *pin;
		else if (*pin != ' ' && *pin != '\t')
		{
			if (cur_token.size() > 0)
			{
				out.push_back(cur_token);
				cur_token = "";
			}
			cur_token = *pin;
			out.push_back(cur_token);
			cur_token = "";
		}
		else if (cur_token.size() > 0)
		{
			out.push_back(cur_token);
			cur_token = "";
		}
		pin++;
	}
	if (cur_token.size() > 0)
		out.push_back(cur_token);
}

static bool token_match(
	const std::vector<std::string>& template_params,
	std::vector<std::string>& template_args,
	std::string* tokens_param, size_t count_tokens_param,
	std::string* tokens_arg, size_t count_tokens_arg)
{
	while (count_tokens_param>0 && count_tokens_param <= count_tokens_arg)
	{
		if (*tokens_param == *tokens_arg)
		{
			tokens_param++;
			tokens_arg++;
			count_tokens_param--;
			count_tokens_arg--;
			continue;
		}
		size_t i = 0;
		for (; i < template_params.size(); i++)
			if (*tokens_param == template_params[i])
				break;
	
		if (i >= template_params.size()) return false;

		std::string templ_arg = *tokens_arg;
		while (count_tokens_param <= count_tokens_arg)
		{
			if (template_args[i] == "" || template_args[i] == templ_arg)
			{
				std::vector<std::string>* cpy = new std::vector<std::string>(template_args);
				(*cpy)[i] = templ_arg;
				bool res = token_match(template_params, *cpy,
					tokens_param + 1, count_tokens_param - 1, tokens_arg + 1, count_tokens_arg - 1);
				if (res)
				{
					template_args = *cpy;
					delete cpy;
					return true;
				}
				delete cpy;
			}
			tokens_arg++;
			count_tokens_arg--;
			if ((*tokens_arg).size() > 1) templ_arg += " ";
			templ_arg += *tokens_arg;			
		}
	}
	return count_tokens_param == 0;
}

static bool s_type_match(const std::vector<std::string>& template_params,
	std::vector<std::string>& template_args, const char* type_param, const char* type_arg)
{
	std::vector<std::string> tokens_param;
	std::vector<std::string> tokens_arg;
	s_tokenize(type_param, tokens_param);
	s_tokenize(type_arg, tokens_arg);

	return token_match(template_params, template_args, tokens_param.data(), tokens_param.size(), tokens_arg.data(), tokens_arg.size());
}

bool TRTCContext::KernelTemplate::deduce_template_args(const DeviceViewable** args, std::vector<std::string>& template_args) const
{
	size_t total = m_template_params.size();
	template_args.resize(total);

	for (size_t i = 0; i < m_type_params.size(); i++)
	{
		std::string type_param = m_type_params[i];
		std::string type_arg = args[i]->name_view_cls();

		bool res = s_type_match(m_template_params, template_args, type_param.c_str(), type_arg.c_str());
		if (!res) return false;
	}
	return true;
}


static std::string cat_templ_args(const std::vector<std::string>& template_args, const TRTCContext* ctx)
{
	std::string ret;
	for (size_t i = 0; i < template_args.size(); i++)
		ret += template_args[i] + ",";

	char str_ptr[32];
	sprintf(str_ptr, "%p", ctx);
	ret += str_ptr;
	return ret;
}

KernelId_t TRTCContext::KernelTemplate::instantiate(TRTCContext& ctx, const std::vector<std::string>& template_args)
{
	std::string templ_pattern = cat_templ_args(template_args, &ctx);

	{
		std::unordered_map<std::string, KernelId_t>::iterator it = m_kernel_id_map.find(templ_pattern);
		if (it != m_kernel_id_map.end()) return it->second;
	}

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

	char md5[33];
	s_get_md5(saxpy.c_str(), md5);

	{
		std::unordered_map<std::string, KernelId_t>::iterator it = ctx.m_kernel_id_map.find(md5);
		if (it != ctx.m_kernel_id_map.end()) 
		{
			m_kernel_id_map[templ_pattern] = it->second;
			return it->second;
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
			if (!ctx._src_to_ptx(saxpy.c_str(), ptx, ptx_size)) return (KernelId_t)(-1);
			
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

	ctx.m_kernel_cache.push_back(kernel);
	KernelId_t ker_id = ctx.m_kernel_cache.size() - 1;
	ctx.m_kernel_id_map[md5] = ker_id;
	m_kernel_id_map[templ_pattern] = ker_id;

	return ker_id;
}

KernelId_t TRTCContext::create_kernel(const std::vector<ParamDesc>& params, const char* body)
{
	KernelTemplate templ({}, params, body);
	return templ.instantiate(*this, {});
}

size_t TRTCContext::get_num_of_params(KernelId_t kernel_id) const
{
	if (kernel_id == (KernelId_t)(-1)) return (size_t)(-1);
	Kernel *kernel = m_kernel_cache[kernel_id];
	return kernel->num_params;
}


void TRTCContext::launch_kernel(KernelId_t kernel_id, dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes) const
{
	if (kernel_id == (KernelId_t)(-1)) return;
	Kernel *kernel = m_kernel_cache[kernel_id];

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


void TRTCContext::launch_kernel_template_explict(KernelTemplate& templ, const std::vector<std::string>& template_args,
	dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes)
{
	KernelId_t kerId = templ.instantiate(*this, template_args);
	launch_kernel(kerId, gridDim, blockDim, args, sharedMemBytes);
}

bool TRTCContext::launch_kernel_template(KernelTemplate& templ, dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes)
{
	std::vector<std::string> template_args;
	if (templ.deduce_template_args(args, template_args))
	{
		size_t total = templ.num_template_params();
		if (template_args.size() >= total)
		{
			size_t i = 0;
			for (; i <total; i++)
				if (template_args[i].size() < 1) break;
			if (i >= total)
			{
				KernelId_t kerId = templ.instantiate(*this, template_args);
				launch_kernel(kerId, gridDim, blockDim, args, sharedMemBytes);
				return true;
			}
		}
	}
	const std::string* type_params = templ.type_params();

	puts("Failed to deduce some of the template arguments.");
	puts("Parameter types:");
	for (size_t i = 0; i < templ.num_params(); i++)
		printf("%s, ", type_params[i].c_str());
	puts("\nArgument types:");
	for (size_t i = 0; i < templ.num_params(); i++)
		printf("%s, ", args[i]->name_view_cls().c_str());
	puts("");

	return false;
}


void TRTCContext::launch_once(dim_type gridDim, dim_type blockDim, const std::vector<AssignedParam>& arg_map, const char* code_body, unsigned sharedMemBytes)
{
	size_t num_params = arg_map.size();
	std::vector<ParamDesc> params(num_params);
	std::vector<std::string> param_types(num_params);
	std::vector<const DeviceViewable*> args(num_params);
	for (size_t i = 0; i < num_params; i++)
	{
		param_types[i] = arg_map[i].arg->name_view_cls();
		params[i] = { param_types[i].c_str(), arg_map[i].param_name };
		args[i] = arg_map[i].arg;
	}
	KernelId_t ker_id = create_kernel(params, code_body);
	launch_kernel(ker_id, gridDim, blockDim, args.data(), sharedMemBytes);
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