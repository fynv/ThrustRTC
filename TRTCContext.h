#ifndef _TRTCContext_h
#define _TRTCContext_h

#include "TRTC_api.h"
#include <vector>
#include <string>
#include <unordered_map>

struct dim_type
{
	unsigned int x, y, z;
};

class THRUST_RTC_API TRTCContext
{
public:
	TRTCContext();
	~TRTCContext();

	static void set_ptx_cache(const char* path);
	void set_verbose(bool verbose = true);

	size_t size_of(const char* cls); // reflect the size of the class in the current context

	class DeviceVector
	{
	public:
		DeviceVector(TRTCContext& ctx, const char* cls, size_t size);
		~DeviceVector();

		std::string get_class() { return m_cls; }
		void* get_data() { return m_data; }

	private:
		std::string m_cls;
		void *m_data;

	};


	struct Kernel;

	struct ParamDesc
	{
		const char* type;
		const char* name;
		std::vector<const char*> template_params;
	};

	class KernelTemplate
	{
	public:
		KernelTemplate(const std::vector<ParamDesc>& params, const char* body, size_t num_reserved_params = 0);
		Kernel* instantiate(const std::unordered_map<std::string, std::string>* template_arg_map) const;


	private:
		std::vector<ParamDesc> m_params;
		std::string m_body;
		size_t m_num_reserved_params;

	};


	void add_include_dir(const char* path);
	void add_built_in_header(const char* name, const char* content);
	void add_inlcude_filename(const char* fn);
	void add_preprocessor(const char* line);	


private:
	bool _src_to_ptx(const char* src, std::vector<char>& ptx);

	std::unordered_map<std::string, size_t> m_size_of_types;


	static const char* s_ptx_cache_path;
	bool m_verbose;

	std::vector<std::string> m_include_dirs;
	std::vector<const char*> m_name_built_in_headers;
	std::vector<const char*> m_content_built_in_headers;
	std::vector<std::string> m_preprocesors;

};


#endif

