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

	struct ParamDesc
	{
		const char* type;
		const char* name;
	};

	class THRUST_RTC_API KernelTemplate
	{
	public:
		KernelTemplate(const std::vector<const char*>& template_params, const std::vector<ParamDesc>& params, const char* body);

		size_t num_template_params() const { return m_template_params.size(); }
		size_t num_params() const { return m_type_params.size(); }
		const std::string* type_params() const { return m_type_params.data(); }

		bool deduce_template_args(const DeviceViewable** args, std::vector<std::string>& template_args) const;
		KernelId_t instantiate(TRTCContext& ctx, const std::vector<std::string>& template_args);

	private:
		std::vector<std::string> m_template_params;
		std::vector<std::string> m_type_params;
		std::string m_code_buf;

		std::unordered_map<std::string, KernelId_t> m_kernel_id_map;
	};

	// create non-templated kernel
	KernelId_t create_kernel(const std::vector<ParamDesc>& params, const char* body);
	size_t get_num_of_params(KernelId_t kernel_id) const;
	void launch_kernel(KernelId_t kernel_id, dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes = 0) const;

	void launch_kernel_template_explict(KernelTemplate& templ, const std::vector<std::string>& template_args,
		dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes = 0);

	bool launch_kernel_template(KernelTemplate& templ, dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes = 0);


	// immediate launch the given code
	struct AssignedParam
	{
		const char* param_name;
		const DeviceViewable* arg;
	};
	void launch_once(dim_type gridDim, dim_type blockDim, const std::vector<AssignedParam>& arg_map, const char* code_body, unsigned sharedMemBytes = 0);

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


#endif

