#ifndef TRTC_for_h
#define TRTC_for_h

#include "TRTC_api.h"
#include "TRTCContext.h"

class THRUST_RTC_API TRTC_For_Template
{
public:
	TRTC_For_Template(const std::vector<const char*>& template_params, const std::vector<TRTCContext::ParamDesc>& params, const char* name_iter, const char* body);
	~TRTC_For_Template();

	size_t num_template_params() const;
	size_t num_params() const;
	const std::string* type_params();

	bool deduce_template_args(const DeviceViewable** args, std::vector<std::string>& template_args) const;
	void launch_explict(TRTCContext& ctx, const std::vector<std::string>& template_args, size_t begin, size_t end, const DeviceViewable** args, unsigned sharedMemBytes = 0);
	bool launch(TRTCContext& ctx, size_t begin, size_t end, const DeviceViewable** args, unsigned sharedMemBytes = 0);

	friend class TRTC_For;

private:
	TRTCContext::KernelTemplate* m_ker_templ;
	
};

class THRUST_RTC_API TRTC_For
{
public:
	TRTC_For(TRTCContext& ctx, TRTC_For_Template& templ, const std::vector<std::string>& template_args);
	TRTC_For(TRTCContext& ctx, const std::vector<TRTCContext::ParamDesc>& params, const char* name_iter, const char* body);

	size_t num_params() const;

	void launch(size_t begin, size_t end, const DeviceViewable** args, unsigned sharedMemBytes = 0) const;

private:
	TRTCContext& m_ctx;
	KernelId_t m_ker_id;
};

void THRUST_RTC_API TRTC_For_Once(TRTCContext& ctx, size_t begin, size_t end, const std::vector<TRTCContext::AssignedParam>& arg_map, const char* name_iter, const char* code_body, unsigned sharedMemBytes = 0);

#endif

