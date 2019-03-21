#include "CachedKernelTemplate.h"

static std::string cat_templ_args(const std::vector<std::string>& template_args)
{
	std::string ret;
	for (size_t i = 0; i < template_args.size(); i++)
		ret += template_args[i] + ",";
	return ret;
}

CachedKernelTemplate::CachedKernelTemplate(const TRTCContext* ctx, 
	const std::vector<const char*> template_params, const std::vector<TRTCContext::ParamDesc>& params,
	const char* body) :m_ctx(ctx), m_templ(template_params, params, body){}

CachedKernelTemplate::~CachedKernelTemplate()
{
	std::unordered_map<std::string, TRTCContext::Kernel*>::iterator it = m_cached_kernels.begin();
	while (it != m_cached_kernels.end())
	{
		TRTCContext::destroy_kernel(it->second);
		it++;
	}
}

void CachedKernelTemplate::launch(dim_type gridDim, dim_type blockDim, const std::vector<std::string>& template_args, DeviceViewable** args, unsigned sharedMemBytes)
{
	std::string key = cat_templ_args(template_args);
	TRTCContext::Kernel* kernel = nullptr;
	std::unordered_map<std::string, TRTCContext::Kernel*>::iterator it = m_cached_kernels.find(key);
	if (it != m_cached_kernels.end())
		kernel = it->second;
	else
	{
		kernel = m_templ.instantiate(*m_ctx, template_args);
		m_cached_kernels[key] = kernel;
	}
	TRTCContext::launch_kernel(kernel, gridDim, blockDim, args, sharedMemBytes);
}

void CachedKernelTemplate::launch(dim_type gridDim, dim_type blockDim, DeviceViewable** args, unsigned sharedMemBytes)
{
	std::vector<std::string> template_args;
	if (m_templ.deduce_template_args(args, template_args))
	{
		size_t total = m_templ.num_template_params();
		if (template_args.size() >= total)
		{
			size_t i = 0;
			for (; i <total; i++)
				if (template_args[i].size() < 1) break;
			if (i >= total)
			{
				launch(gridDim, blockDim, template_args, args, sharedMemBytes);
				return;
			}
		}
	}

	const std::string* type_params = m_templ.type_params();

	puts("Failed to deduce some of the template arguments.");
	puts("Parameter types:");
	for (size_t i = 0; i < m_templ.num_params(); i++)
		printf("%s, ", type_params[i].c_str());
	puts("\nArgument types:");
	for (size_t i = 0; i < m_templ.num_params(); i++)
		printf("%s, ", args[i]->name_view_cls().c_str());
	puts("");
}
