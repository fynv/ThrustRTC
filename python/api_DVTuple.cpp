#include "api.h"
#include "TRTCContext.h"
#include "DVTuple.h"

typedef std::vector<std::string> StrArray;
typedef std::vector<const DeviceViewable*> PtrArray;

void* n_dvtuple_create(void* ptr_dvs, void* ptr_names)
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
	return new DVTuple(arg_map);
}

