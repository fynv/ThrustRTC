#include "api.h"
#include "TRTCContext.h"
#include "functor.h"

typedef std::vector<std::string> StrArray;
typedef std::vector<const DeviceViewable*> PtrArray;


void* n_functor_create(void* ptr_dvs, void* ptr_names, void* ptr_functor_params, const char* code_body)
{
	PtrArray* dvs = (PtrArray*)ptr_dvs;
	StrArray* names = (StrArray*)ptr_names;
	size_t num_params = dvs->size();
	std::vector<CapturedDeviceViewable> arg_map(num_params);
	for (size_t i = 0; i < num_params; i++)
	{
		arg_map[i].obj_name = (*names)[i].c_str();
		arg_map[i].obj = (*dvs)[i];
	}

	StrArray* str_functor_params = (StrArray*)ptr_functor_params;
	size_t num_functor_params = str_functor_params->size();
	std::vector<const char*> functor_params(num_functor_params);
	for (size_t i = 0; i < num_functor_params; i++)
		functor_params[i] = (*str_functor_params)[i].c_str();

	return new Functor(arg_map, functor_params, code_body);
}

void* n_built_in_functor_create(const char* name_built_in_view_cls)
{
	return new Functor(name_built_in_view_cls);
}

