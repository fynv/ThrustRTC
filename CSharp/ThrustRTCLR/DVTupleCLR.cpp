
#include "ThrustRTCLR.h"
#include "DVTuple.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	IntPtr Native::dvtuple_create(array<CapturedDeviceViewable_clr>^ p_elem_map)
	{
		int num_params = p_elem_map->Length;
		std::vector<CapturedDeviceViewable> arg_map(num_params);
		for (int i = 0; i < num_params; i++)
		{
			arg_map[i].obj_name = just_cast_it<const char>(p_elem_map[i].obj_name);
			arg_map[i].obj = just_cast_it<DeviceViewable>(p_elem_map[i].obj);
		}
		return (IntPtr)(new DVTuple(arg_map));
	}

}
