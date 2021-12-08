#ifndef _TRTCContext_h
#define _TRTCContext_h

#include "TRTC_api.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "DeviceViewable.h"

struct dim_type
{
	unsigned int x, y, z;
};

void THRUST_RTC_API set_libnvrtc_path(const char* path);

bool THRUST_RTC_API TRTC_Try_Init();

int THRUST_RTC_API TRTC_Get_PTX_Arch();

void THRUST_RTC_API TRTC_Set_Verbose(bool verbose = true);
void THRUST_RTC_API TRTC_Set_Kernel_Debug(bool kernel_debug = true);

// reflection 
size_t THRUST_RTC_API TRTC_Size_Of(const char* cls);
bool THRUST_RTC_API TRTC_Query_Struct(const char* name_struct, const std::vector<const char*>& name_members, size_t* offsets);

// Adding definitions to device code
void THRUST_RTC_API TRTC_Add_Include_Dir(const char* path);
void THRUST_RTC_API TRTC_Add_Built_In_Header(const char* name, const char* content);
void THRUST_RTC_API TRTC_Add_Code_Block(const char* code);
void THRUST_RTC_API TRTC_Add_Inlcude_Filename(const char* fn);
void THRUST_RTC_API TRTC_Add_Constant_Object(const char* name, const DeviceViewable& obj);
std::string THRUST_RTC_API TRTC_Add_Struct(const char* struct_body);

void THRUST_RTC_API TRTC_Wait();

class THRUST_RTC_API TRTC_Kernel
{
public:
	size_t num_params() const { return m_param_names.size();  }

	TRTC_Kernel(const std::vector<const char*>& param_names, const char* code_body);
	bool calc_optimal_block_size(const DeviceViewable** args, int& sizeBlock, unsigned sharedMemBytes = 0);
	bool calc_number_blocks(const DeviceViewable** args, int sizeBlock, int& numBlocks, unsigned sharedMemBytes = 0);
	bool launch(dim_type gridDim, dim_type blockDim, const DeviceViewable** args, unsigned sharedMemBytes = 0);

private:
	std::vector<std::string> m_param_names;
	std::string m_code_body;

};


class THRUST_RTC_API TRTC_For
{
public:
	size_t num_params() const { return m_param_names.size(); }

	TRTC_For(const std::vector<const char*>& param_names, const char* name_iter, const char* code_body);
	bool launch(size_t begin, size_t end, const DeviceViewable** args);
	bool launch_n(size_t n, const DeviceViewable** args);

private:
	std::vector<std::string> m_param_names;
	std::string m_name_iter;
	std::string m_code_body;
};


#endif

