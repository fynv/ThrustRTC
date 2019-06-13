#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "Functor.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	IntPtr Native::functor_create(array<CapturedDeviceViewable_clr>^ p_arg_map, array<IntPtr>^ p_functor_params, IntPtr p_code_body)
	{
		int num_params = p_arg_map->Length;
		std::vector<CapturedDeviceViewable> arg_map(num_params);
		for (int i = 0; i < num_params; i++)
		{
			arg_map[i].obj_name = just_cast_it<const char>(p_arg_map[i].obj_name);
			arg_map[i].obj = just_cast_it<DeviceViewable>(p_arg_map[i].obj);
		}
		int num_functor_params = p_functor_params->Length;
		std::vector<const char*> functor_params(num_functor_params);
		for (int i = 0; i < num_functor_params; i++)
			functor_params[i] = just_cast_it<const char>(p_functor_params[i]);
		const char* code_body = just_cast_it<const char>(p_code_body);
		return (IntPtr)(new Functor(arg_map, functor_params, code_body));
	}

	IntPtr Native::built_in_functor_create(IntPtr p_name_built_in_view_cls)
	{
		const char* name_built_in_view_cls = just_cast_it<const char>(p_name_built_in_view_cls);
		return (IntPtr)(new Functor(name_built_in_view_cls));
	}

}