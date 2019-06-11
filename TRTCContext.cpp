#include "TRTCContext.h"
#include <string>
#include <string.h>
#include <stdio.h>
#include <unqlite.h>
#include "cuda_wrapper.h"
#include "nvtrc_wrapper.h"
#include "launch_calc.h"
#include "crc64.h"
#include "functor.h"
#include "cuda_inline_headers.hpp"
#include "cuda_inline_headers_global.hpp"

typedef unsigned int KernelId_t;

class TRTCContext
{
public:
	static void set_libnvrtc_path(const char* path);
	static TRTCContext& get_context();

	void set_verbose(bool verbose = true);

	// reflection 
	size_t size_of(const char* cls);
	bool query_struct(const char* name_struct, const std::vector<const char*>& name_members, size_t* offsets);
	bool calc_optimal_block_size(const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body, int& sizeBlock, unsigned sharedMemBytes = 0);
	bool calc_number_blocks(const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body, int sizeBlock, int& numBlocks, unsigned sharedMemBytes = 0);
	bool launch_kernel(dim_type gridDim, dim_type blockDim, const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body, unsigned sharedMemBytes = 0);
	bool launch_for(size_t begin, size_t end, const std::vector<CapturedDeviceViewable>& arg_map, const char* name_iter, const char* code_body);
	bool launch_for_n(size_t n, const std::vector<CapturedDeviceViewable>& arg_map, const char* name_iter, const char* code_body);

	void add_include_dir(const char* path);
	void add_built_in_header(const char* name, const char* content);
	void add_code_block(const char* code);
	void add_inlcude_filename(const char* fn);
	void add_constant_object(const char* name, const DeviceViewable& obj);
	std::string add_struct(const char* struct_body);

private:
	TRTCContext();
	~TRTCContext();

	bool _src_to_ptx(const char* src, std::vector<char>& ptx, size_t& ptx_size) const;
	KernelId_t _build_kernel(const std::vector<CapturedDeviceViewable>& arg_map, const char* code_body);
	int _launch_calc(KernelId_t kid, unsigned sharedMemBytes);
	int _persist_calc(KernelId_t kid, int numBlocks, unsigned sharedMemBytes);
	bool _launch_kernel(KernelId_t kid, dim_type gridDim, dim_type blockDim, const std::vector<CapturedDeviceViewable>& arg_map, unsigned sharedMemBytes);

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

#include "impl_context.inl"

void set_libnvrtc_path(const char* path)
{
	TRTCContext::set_libnvrtc_path(path);
}

void TRTC_Set_Verbose(bool verbose)
{
	TRTCContext& ctx = TRTCContext::get_context();
	ctx.set_verbose(verbose);
}

size_t TRTC_Size_Of(const char* cls)
{
	TRTCContext& ctx = TRTCContext::get_context();
	return ctx.size_of(cls);
}

bool TRTC_Query_Struct(const char* name_struct, const std::vector<const char*>& name_members, size_t* offsets)
{
	TRTCContext& ctx = TRTCContext::get_context();
	return ctx.query_struct(name_struct, name_members, offsets);
}

void TRTC_Add_Include_Dir(const char* path)
{
	TRTCContext& ctx = TRTCContext::get_context();
	ctx.add_include_dir(path);
}

void TRTC_Add_Built_In_Header(const char* name, const char* content)
{
	TRTCContext& ctx = TRTCContext::get_context();
	ctx.add_built_in_header(name, content);
}

void TRTC_Add_Code_Block(const char* code)
{
	TRTCContext& ctx = TRTCContext::get_context();
	ctx.add_code_block(code);
}

void TRTC_Add_Inlcude_Filename(const char* fn)
{
	TRTCContext& ctx = TRTCContext::get_context();
	ctx.add_inlcude_filename(fn);
}

void TRTC_Add_Constant_Object(const char* name, const DeviceViewable& obj)
{
	TRTCContext& ctx = TRTCContext::get_context();
	ctx.add_constant_object(name, obj);
}

std::string TRTC_Add_Struct(const char* struct_body)
{
	TRTCContext& ctx = TRTCContext::get_context();
	return ctx.add_struct(struct_body);
}

TRTC_Kernel::TRTC_Kernel(const std::vector<const char*>& param_names, const char* code_body) :
m_param_names(param_names.size()), m_code_body(code_body)
{
	for (size_t i = 0; i < param_names.size(); i++)
		m_param_names[i] = param_names[i];
}

bool TRTC_Kernel::calc_optimal_block_size(const DeviceViewable** args, int& sizeBlock, unsigned sharedMemBytes)
{
	TRTCContext& ctx = TRTCContext::get_context();
	std::vector<CapturedDeviceViewable> arg_map(m_param_names.size());
	for (size_t i = 0; i < m_param_names.size(); i++)
	{
		arg_map[i].obj_name = m_param_names[i].c_str();
		arg_map[i].obj = args[i];
	}
	return ctx.calc_optimal_block_size(arg_map, m_code_body.c_str(), sizeBlock, sharedMemBytes);
}

bool TRTC_Kernel::calc_number_blocks(const DeviceViewable** args, int sizeBlock, int& numBlocks, unsigned sharedMemBytes)
{
	TRTCContext& ctx = TRTCContext::get_context();
	std::vector<CapturedDeviceViewable> arg_map(m_param_names.size());
	for (size_t i = 0; i < m_param_names.size(); i++)
	{
		arg_map[i].obj_name = m_param_names[i].c_str();
		arg_map[i].obj = args[i];
	}
	return ctx.calc_number_blocks(arg_map, m_code_body.c_str(), sizeBlock, numBlocks, sharedMemBytes);
}

bool TRTC_Kernel::launch(dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes)
{
	TRTCContext& ctx = TRTCContext::get_context();
	std::vector<CapturedDeviceViewable> arg_map(m_param_names.size());
	for (size_t i = 0; i < m_param_names.size(); i++)
	{
		arg_map[i].obj_name = m_param_names[i].c_str();
		arg_map[i].obj = args[i];
	}
	return ctx.launch_kernel(gridDim, blockDim, arg_map, m_code_body.c_str(), sharedMemBytes);
}

TRTC_For::TRTC_For(const std::vector<const char*>& param_names, const char* name_iter, const char* code_body) :
m_param_names(param_names.size()), m_name_iter(name_iter), m_code_body(code_body)
{
	for (size_t i = 0; i < param_names.size(); i++)
		m_param_names[i] = param_names[i];
}

bool TRTC_For::launch(size_t begin, size_t end, const DeviceViewable** args)
{
	TRTCContext& ctx = TRTCContext::get_context();
	std::vector<CapturedDeviceViewable> arg_map(m_param_names.size());
	for (size_t i = 0; i < m_param_names.size(); i++)
	{
		arg_map[i].obj_name = m_param_names[i].c_str();
		arg_map[i].obj = args[i];
	}
	return ctx.launch_for(begin, end, arg_map, m_name_iter.c_str(), m_code_body.c_str());
}


bool TRTC_For::launch_n(size_t n, const DeviceViewable** args)
{
	TRTCContext& ctx = TRTCContext::get_context();
	std::vector<CapturedDeviceViewable> arg_map(m_param_names.size());
	for (size_t i = 0; i < m_param_names.size(); i++)
	{
		arg_map[i].obj_name = m_param_names[i].c_str();
		arg_map[i].obj = args[i];
	}
	return ctx.launch_for_n(n, arg_map, m_name_iter.c_str(), m_code_body.c_str());
}
