#include "api.h"
#include "TRTCContext.h"
#include <string>
#include <vector>

typedef std::vector<std::string> StrArray;
typedef std::vector<const DeviceViewable*> PtrArray;

void n_set_libnvrtc_path(const char* path)
{
	set_libnvrtc_path(path);
}

int n_trtc_try_init()
{
	return TRTC_Try_Init() ? 0 : -100;
}

void n_set_verbose(unsigned verbose)
{
	TRTC_Set_Verbose(verbose!=0);
}

void n_add_include_dir(const char* dir)
{
	TRTC_Add_Include_Dir(dir);
}

void n_add_built_in_header(const char* filename, const char* filecontent)
{
	TRTC_Add_Built_In_Header(filename, filecontent);
}

void n_add_inlcude_filename(const char* fn)
{
	TRTC_Add_Inlcude_Filename(fn);
}

void n_add_code_block(const char* line)
{
	TRTC_Add_Code_Block(line);
}

void n_add_constant_object(const char* name, void* cptr)
{
	DeviceViewable* dv = (DeviceViewable*)cptr;
	TRTC_Add_Constant_Object(name, *dv);
}

void n_wait()
{
	TRTC_Wait();
}

void* n_kernel_create(void* ptr_param_list, const char* body)
{
	StrArray* param_list = (StrArray*)ptr_param_list;
	size_t num_params = param_list->size();
	std::vector<const char*> params(num_params);
	for (size_t i = 0; i < num_params; i++)
		params[i] = (*param_list)[i].c_str();	
	TRTC_Kernel* cptr = new TRTC_Kernel(params, body);
	return cptr;
}

void n_kernel_destroy(void* cptr)
{
	TRTC_Kernel* kernel = (TRTC_Kernel*)cptr;
	delete kernel;
}

int n_kernel_num_params(void* cptr)
{
	TRTC_Kernel* kernel = (TRTC_Kernel*)cptr;
	return (int)kernel->num_params();
}


int n_kernel_calc_optimal_block_size(void* ptr_kernel, void* ptr_arg_list, unsigned sharedMemBytes)
{
	TRTC_Kernel* kernel = (TRTC_Kernel*)ptr_kernel;
	size_t num_params = kernel->num_params();
	PtrArray* arg_list = (PtrArray*)ptr_arg_list;

	size_t size = arg_list->size();
	if (num_params != size)
	{
		printf("Wrong number of arguments received. %d required, %d received.", (int)num_params, (int)size);
		return -2;
	}
	
	int sizeBlock;
	if (kernel->calc_optimal_block_size(arg_list->data(), sizeBlock, sharedMemBytes))
		return sizeBlock;
	else
	{
		printf("Failed to calculate optimal block size.\n");
		return -1;
	}
}


int n_kernel_calc_number_blocks(void* ptr_kernel, void* ptr_arg_list, int sizeBlock, unsigned sharedMemBytes)
{
	TRTC_Kernel* kernel = (TRTC_Kernel*)ptr_kernel;
	size_t num_params = kernel->num_params();
	PtrArray* arg_list = (PtrArray*)ptr_arg_list;

	size_t size = arg_list->size();	
	if (num_params != size)
	{
		printf("Wrong number of arguments received. %d required, %d received.", (int)num_params, (int)size);
		return -2;
	}
	
	int numBlocks;
	if (kernel->calc_number_blocks(arg_list->data(), sizeBlock, numBlocks, sharedMemBytes))
		return numBlocks;
	else
	{
		printf("Failed to calculate number of persistant blocks.\n");
		return -1;
	}
}

int n_kernel_launch(void* ptr_kernel, void* ptr_gridDim, void* ptr_blockDim, void* ptr_arg_list, int sharedMemBytes)
{
	TRTC_Kernel* kernel = (TRTC_Kernel*)ptr_kernel;
	size_t num_params = kernel->num_params();

	dim_type* gridDim = (dim_type*)ptr_gridDim;
	dim_type* blockDim = (dim_type*)ptr_blockDim;	

	PtrArray* arg_list = (PtrArray*)ptr_arg_list;

	size_t size = arg_list->size();
	if (num_params != size)
	{
		printf("Wrong number of arguments received. %d required, %d received.", (int)num_params, (int)size);
		return -2;
	}

	if (kernel->launch(*gridDim, *blockDim, arg_list->data(), sharedMemBytes))
		return 0;
	else
		return -1;
}

void* n_for_create(void* ptr_param_list, const char* name_iter, const char* body)
{
	StrArray* param_list = (StrArray*)ptr_param_list;
	size_t num_params = param_list->size();
	std::vector<const char*> params(num_params);
	for (size_t i = 0; i < num_params; i++)
		params[i] = (*param_list)[i].c_str();	
	TRTC_For* cptr = new TRTC_For(params, name_iter, body);
	return cptr;
}


void n_for_destroy(void* cptr)
{
	TRTC_For* p_for = (TRTC_For*)cptr;
	delete p_for;
}

int n_for_num_params(void* cptr)
{
	TRTC_For* p_for = (TRTC_For*)cptr;
	return (int)p_for->num_params();
}

int n_for_launch(void* cptr, int begin, int end, void* ptr_arg_list)
{
	TRTC_For* p_for = (TRTC_For*)cptr;
	size_t num_params = p_for->num_params();

	PtrArray* arg_list = (PtrArray*)ptr_arg_list;

	size_t size = arg_list->size();
	if (num_params != size)
	{
		printf("Wrong number of arguments received. %d required, %d received.", (int)num_params, (int)size);
		return -2;
	}

	if (p_for->launch(begin, end, arg_list->data()))
		return 0;
	else
		return -1;
}

int n_for_launch_n(void* cptr, int n, void* ptr_arg_list)
{
	TRTC_For* p_for = (TRTC_For*)cptr;
	size_t num_params = p_for->num_params();

	PtrArray* arg_list = (PtrArray*)ptr_arg_list;

	size_t size = arg_list->size();
	if (num_params != size)
	{
		printf("Wrong number of arguments received. %d required, %d received.", (int)num_params, (int)size);
		return -2;
	}

	if (p_for->launch_n(n, arg_list->data()))
		return 0;
	else
		return -1;
}
