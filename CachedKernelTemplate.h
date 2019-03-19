#ifndef _CachedKernelTemplate_h
#define _CachedKernelTemplate_h

#include <unordered_map>
#include "TRTC_api.h"
#include "TRTCContext.h"

class THRUST_RTC_API CachedKernelTemplate
{
public:
	CachedKernelTemplate(const TRTCContext* ctx, const std::vector<TRTCContext::ParamDesc>& params, const char* body, const std::vector<const char*> template_params = {});
	~CachedKernelTemplate();

	size_t num_params() const { return m_templ.num_params(); }
	size_t num_template_params() const { return m_templ.num_template_params(); }

	// explicit
	void launch(dim_type gridDim, dim_type blockDim, DeviceViewable** args, const std::vector<std::string>& template_args, unsigned sharedMemBytes = 0);

	// deduce
	void launch(dim_type gridDim, dim_type blockDim, DeviceViewable** args, unsigned sharedMemBytes = 0);

private:
	std::unordered_map<std::string, TRTCContext::Kernel*> m_cached_kernels;

	const TRTCContext* m_ctx;
	TRTCContext::KernelTemplate m_templ;

};

#endif
