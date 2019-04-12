#ifndef _TRTCContext_h
#define _TRTCContext_h

#include <vector>
#include <string>
#include <unordered_map>

#include "TRTC_api.h"
#include "DeviceViewable.h"

struct dim_type
{
	unsigned int x, y, z;
};

typedef unsigned int KernelId_t;

class THRUST_RTC_API TRTCContext
{
public:
	TRTCContext();
	~TRTCContext();

	static void set_ptx_cache(const char* path);
	void set_verbose(bool verbose = true);

	size_t size_of(const char* cls); // reflect the size of the class in the current context

	struct AssignedParam
	{
		const char* param_name;
		const DeviceViewable* arg;
	};

	bool launch_kernel(dim_type gridDim, dim_type blockDim, const std::vector<AssignedParam>& arg_map, const char* code_body, unsigned sharedMemBytes = 0);
	bool launch_for(size_t begin, size_t end, const std::vector<TRTCContext::AssignedParam>& arg_map, const char* name_iter, const char* code_body, unsigned sharedMemBytes = 0);
	
	void add_include_dir(const char* path);
	void add_built_in_header(const char* name, const char* content);
	void add_inlcude_filename(const char* fn);
	void add_preprocessor(const char* line);	
	void add_constant_object(const char* name, const DeviceViewable& obj);

private:
	bool _src_to_ptx(const char* src, std::vector<char>& ptx, size_t& ptx_size) const;

	static const char* s_ptx_cache_path;

	bool m_verbose;
	std::vector<std::string> m_include_dirs;
	std::vector<const char*> m_name_built_in_headers;
	std::vector<const char*> m_content_built_in_headers;
	std::vector<std::string> m_preprocesors;
	std::vector<std::pair<std::string, ViewBuf>> m_constants;

	std::unordered_map<std::string, size_t> m_size_of_types;

	struct Kernel;
	std::vector<Kernel*> m_kernel_cache;
	std::unordered_map<std::string, KernelId_t> m_kernel_id_map;

};


class THRUST_RTC_API TRTC_Kernel
{
public:
	size_t num_params() const { return m_param_names.size();  }

	TRTC_Kernel(const std::vector<const char*>& param_names, const char* code_body);
	bool launch(TRTCContext& ctx, dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes = 0);

private:
	std::vector<std::string> m_param_names;
	std::string m_code_body;

};


class THRUST_RTC_API TRTC_For
{
public:
	size_t num_params() const { return m_param_names.size(); }

	TRTC_For(const std::vector<const char*>& param_names, const char* name_iter, const char* code_body);
	bool launch(TRTCContext& ctx, size_t begin, size_t end, const DeviceViewable** args, unsigned sharedMemBytes = 0);

private:
	std::vector<std::string> m_param_names;
	std::string m_name_iter;
	std::string m_code_body;
};


#endif

