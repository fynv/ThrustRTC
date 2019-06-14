#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "fake_vectors/DVConstant.h"
#include "fake_vectors/DVCounter.h"
#include "fake_vectors/DVDiscard.h"
#include "fake_vectors/DVPermutation.h"
#include "fake_vectors/DVReverse.h"
#include "fake_vectors/DVTransform.h"
#include "fake_vectors/DVZipped.h"
#include "fake_vectors/DVCustomVector.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	IntPtr Native::dvconstant_create(IntPtr p_dvobj, size_t size)
	{
		DeviceViewable* dvobj = just_cast_it<DeviceViewable>(p_dvobj);
		return (IntPtr)(new DVConstant(*dvobj, size));
	}

	IntPtr Native::dvcounter_create(IntPtr p_dvobj_init, size_t size)
	{
		DeviceViewable* dvobj_init = just_cast_it<DeviceViewable>(p_dvobj_init);
		return (IntPtr)(new DVCounter(*dvobj_init, size));
	}

	IntPtr Native::dvdiscard_create(IntPtr p_elem_cls, size_t size)
	{
		const char* elem_cls = just_cast_it<const char>(p_elem_cls);
		return (IntPtr)(new DVDiscard(elem_cls, size));
	}

	IntPtr Native::dvpermutation_create(IntPtr p_vec_value, IntPtr p_vec_index)
	{
		DVVectorLike* vec_value = just_cast_it<DVVectorLike>(p_vec_value);
		DVVectorLike* vec_index = just_cast_it<DVVectorLike>(p_vec_index);
		return (IntPtr)(new DVPermutation(*vec_value, *vec_index));
	}

	IntPtr Native::dvreverse_create(IntPtr p_vec_value)
	{
		DVVectorLike* vec_value = just_cast_it<DVVectorLike>(p_vec_value);
		return (IntPtr)(new DVReverse(*vec_value));
	}

	IntPtr Native::dvtransform_create(IntPtr p_vec_in, IntPtr p_elem_cls, IntPtr p_op)
	{
		DVVectorLike* vec_in = just_cast_it<DVVectorLike>(p_vec_in);
		const char* elem_cls = just_cast_it<const char>(p_elem_cls);
		Functor* op = just_cast_it<Functor>(p_op);
		return (IntPtr)(new DVTransform(*vec_in, elem_cls, *op));
	}

	IntPtr Native::dvzipped_create(array<IntPtr>^ p_vecs, array<IntPtr>^ p_elem_names)
	{
		int num_vecs = p_vecs->Length;
		std::vector<DVVectorLike*> vecs(num_vecs);
		for (int i = 0; i < num_vecs; i++)
			vecs[i] = just_cast_it<DVVectorLike>(p_vecs[i]);

		int num_elems = p_elem_names->Length;
		if (num_elems != num_vecs) return IntPtr::Zero;
		std::vector<const char*> elem_names(num_elems);
		for (int i = 0; i < num_elems; i++)
			elem_names[i] = just_cast_it<const char>(p_elem_names[i]);

		return (IntPtr)(new DVZipped(vecs, elem_names));
	}

	IntPtr Native::dvcustomvector_create(array<CapturedDeviceViewable_clr>^ p_arg_map, IntPtr p_name_idx, IntPtr p_code_body, IntPtr p_elem_cls, size_t size, bool read_only)
	{
		int num_params = p_arg_map->Length;
		std::vector<CapturedDeviceViewable> arg_map(num_params);
		for (int i = 0; i < num_params; i++)
		{
			arg_map[i].obj_name = just_cast_it<const char>(p_arg_map[i].obj_name);
			arg_map[i].obj = just_cast_it<DeviceViewable>(p_arg_map[i].obj);
		}
		
		const char* name_idx = just_cast_it<const char>(p_name_idx);
		const char* code_body = just_cast_it<const char>(p_code_body);
		const char* elem_cls = just_cast_it<const char>(p_elem_cls);

		return (IntPtr)(new DVCustomVector(arg_map, name_idx, code_body, elem_cls, size, read_only));
	}

}