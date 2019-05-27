#ifndef _TRTCContext_h
#define _TRTCContext_h

#pragma warning( disable: 4251 )

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

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

	static void set_libnvrtc_path(const char* path);

	void set_verbose(bool verbose = true);

	// reflection 
	size_t size_of(const char* cls);
	bool query_struct(const char* name_struct, const std::vector<const char*>& name_members, size_t* offsets);

	struct AssignedParam
	{
		const char* param_name;
		const DeviceViewable* arg;
	};

	bool calc_optimal_block_size(const std::vector<AssignedParam>& arg_map, const char* code_body, int& sizeBlock, unsigned sharedMemBytes = 0);
	bool calc_number_blocks(const std::vector<AssignedParam>& arg_map, const char* code_body, int sizeBlock, int& numBlocks, unsigned sharedMemBytes = 0);
	bool launch_kernel(dim_type gridDim, dim_type blockDim, const std::vector<AssignedParam>& arg_map, const char* code_body, unsigned sharedMemBytes = 0);
	bool launch_for(size_t begin, size_t end, const std::vector<TRTCContext::AssignedParam>& arg_map, const char* name_iter, const char* code_body);
	bool launch_for_n(size_t n, const std::vector<TRTCContext::AssignedParam>& arg_map, const char* name_iter, const char* code_body);
	
	void add_include_dir(const char* path);
	void add_built_in_header(const char* name, const char* content);
	void add_code_block(const char* code);
	void add_inlcude_filename(const char* fn);
	void add_constant_object(const char* name, const DeviceViewable& obj);
	std::string add_struct(const char* struct_body);

private:
	bool _src_to_ptx(const char* src, std::vector<char>& ptx, size_t& ptx_size) const;
	KernelId_t _build_kernel(const std::vector<AssignedParam>& arg_map, const char* code_body);
	int _launch_calc(KernelId_t kid, unsigned sharedMemBytes);
	int _persist_calc(KernelId_t kid, int numBlocks, unsigned sharedMemBytes);
	bool _launch_kernel(KernelId_t kid, dim_type gridDim, dim_type blockDim, const std::vector<AssignedParam>& arg_map, unsigned sharedMemBytes);

	static const char* s_libnvrtc_path;

	bool m_verbose;
	std::vector<std::string> m_include_dirs;
	std::vector<const char*> m_name_built_in_headers;
	std::vector<const char*> m_content_built_in_headers;
	std::vector<std::string> m_code_blocks;
	std::vector<std::pair<std::string, ViewBuf>> m_constants;

	std::string m_header_of_structs;
	std::string m_name_header_of_structs;
	std::unordered_set<int64_t> m_known_structs;

	std::unordered_map<std::string, size_t> m_size_of_types;
	std::unordered_map<std::string, std::vector<size_t>> m_offsets_of_structs;

	struct Kernel;
	std::vector<Kernel*> m_kernel_cache;
	std::unordered_map<int64_t, KernelId_t> m_kernel_id_map;
};


class THRUST_RTC_API TRTC_Kernel
{
public:
	size_t num_params() const { return m_param_names.size();  }

	TRTC_Kernel(const std::vector<const char*>& param_names, const char* code_body);
	bool calc_optimal_block_size(TRTCContext& ctx, const DeviceViewable** args, int& sizeBlock, unsigned sharedMemBytes = 0);
	bool calc_number_blocks(TRTCContext& ctx, const DeviceViewable** args, int sizeBlock, int& numBlocks, unsigned sharedMemBytes = 0);
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
	bool launch(TRTCContext& ctx, size_t begin, size_t end, const DeviceViewable** args);
	bool launch_n(TRTCContext& ctx, size_t n, const DeviceViewable** args);

private:
	std::vector<std::string> m_param_names;
	std::string m_name_iter;
	std::string m_code_body;
};


#endif

