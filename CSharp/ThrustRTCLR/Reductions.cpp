#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "count.h"
#include "reduce.h"
#include "equal.h"
#include "extrema.h"
#include "inner_product.h"
#include "transform_reduce.h"
#include "logical.h"
#include "partition.h"
#include "sort.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	size_t Native::count(IntPtr p_vec, IntPtr p_value)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable* value = just_cast_it<DeviceViewable>(p_value);
		size_t ret;
		if (TRTC_Count(*vec, *value, ret))
			return ret;
		else
			return (size_t)(-1);
	}

	size_t Native::count_if(IntPtr p_vec, IntPtr p_pred)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		size_t ret;
		if (TRTC_Count_If(*vec, *pred, ret))
			return ret;
		else
			return (size_t)(-1);
	}

	static Object^ s_box_basic_type(const ViewBuf& v, const std::string& type)
	{
		if (type == "int8_t") return *(int8_t*)v.data();
		if (type == "uint8_t") return *(uint8_t*)v.data();
		if (type == "int16_t") return *(int16_t*)v.data();
		if (type == "uint16_t") return *(uint16_t*)v.data();
		if (type == "int32_t") return *(int32_t*)v.data();
		if (type == "uint32_t") return *(uint32_t*)v.data();
		if (type == "int64_t") return *(int64_t*)v.data();
		if (type == "uint64_t") return *(uint64_t*)v.data();
		if (type == "float") return *(float*)v.data();
		if (type == "double") return *(double*)v.data();
		if (type == "bool") return *(bool*)v.data();
		return nullptr;
	}

	Object^ Native::reduce(IntPtr p_vec)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		ViewBuf ret;
		if (!TRTC_Reduce(*vec, ret)) return nullptr;
		return s_box_basic_type(ret, vec->name_elem_cls());
	}

	Object^ Native::reduce(IntPtr p_vec, IntPtr p_init)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable *init = just_cast_it<DeviceViewable>(p_init);
		ViewBuf ret;
		if (!TRTC_Reduce(*vec, *init, ret)) return nullptr;
		return s_box_basic_type(ret, vec->name_elem_cls());
	}

	Object^ Native::reduce(IntPtr p_vec, IntPtr p_init, IntPtr p_binary_op)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		DeviceViewable *init = just_cast_it<DeviceViewable>(p_init);
		Functor* binary_op = just_cast_it<Functor>(p_binary_op);
		ViewBuf ret;
		if (!TRTC_Reduce(*vec, *init, *binary_op, ret)) return nullptr;
		return s_box_basic_type(ret, vec->name_elem_cls());
	}

	uint32_t Native::reduce_by_key(IntPtr p_key_in, IntPtr p_value_in, IntPtr p_key_out, IntPtr p_value_out)
	{
		DVVectorLike* key_in = just_cast_it<DVVectorLike>(p_key_in);
		DVVectorLike* value_in = just_cast_it<DVVectorLike>(p_value_in);
		DVVectorLike* key_out = just_cast_it<DVVectorLike>(p_key_out);
		DVVectorLike* value_out = just_cast_it<DVVectorLike>(p_value_out);
		return TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out);
	}

	uint32_t Native::reduce_by_key(IntPtr p_key_in, IntPtr p_value_in, IntPtr p_key_out, IntPtr p_value_out, IntPtr p_binary_pred)
	{
		DVVectorLike* key_in = just_cast_it<DVVectorLike>(p_key_in);
		DVVectorLike* value_in = just_cast_it<DVVectorLike>(p_value_in);
		DVVectorLike* key_out = just_cast_it<DVVectorLike>(p_key_out);
		DVVectorLike* value_out = just_cast_it<DVVectorLike>(p_value_out);
		Functor* binary_pred = just_cast_it<Functor>(p_binary_pred);
		return TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out, *binary_pred);
	}

	uint32_t Native::reduce_by_key(IntPtr p_key_in, IntPtr p_value_in, IntPtr p_key_out, IntPtr p_value_out, IntPtr p_binary_pred, IntPtr p_binary_op)
	{
		DVVectorLike* key_in = just_cast_it<DVVectorLike>(p_key_in);
		DVVectorLike* value_in = just_cast_it<DVVectorLike>(p_value_in);
		DVVectorLike* key_out = just_cast_it<DVVectorLike>(p_key_out);
		DVVectorLike* value_out = just_cast_it<DVVectorLike>(p_value_out);
		Functor* binary_pred = just_cast_it<Functor>(p_binary_pred);
		Functor* binary_op = just_cast_it<Functor>(p_binary_op);
		return TRTC_Reduce_By_Key(*key_in, *value_in, *key_out, *value_out, *binary_pred, *binary_op);

	}

	Object^ Native::equal(IntPtr p_vec1, IntPtr p_vec2)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		bool ret;
		if (!TRTC_Equal(*vec1, *vec2, ret))
			return nullptr;
		return ret;
	}

	Object^ Native::equal(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_binary_pred)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		Functor* binary_pred = just_cast_it<Functor>(p_binary_pred);
		bool ret;
		if (!TRTC_Equal(*vec1, *vec2, *binary_pred, ret))
			return nullptr;
		return ret;
	}

	size_t Native::min_element(IntPtr p_vec)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		size_t id_min;
		if (!TRTC_Min_Element(*vec, id_min))
			return (size_t)(-1);
		return id_min;
	}

	size_t Native::min_element(IntPtr p_vec, IntPtr p_comp)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* comp = just_cast_it<Functor>(p_comp);
		size_t id_min;
		if (!TRTC_Min_Element(*vec, *comp, id_min))
			return (size_t)(-1);
		return id_min;
	}

	size_t Native::max_element(IntPtr p_vec)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		size_t id_max;
		if (!TRTC_Max_Element(*vec, id_max))
			return (size_t)(-1);
		return id_max;
	}

	size_t Native::max_element(IntPtr p_vec, IntPtr p_comp)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* comp = just_cast_it<Functor>(p_comp);
		size_t id_max;
		if (!TRTC_Max_Element(*vec, *comp, id_max))
			return (size_t)(-1);
		return id_max;
	}

	Tuple<int64_t, int64_t>^ Native::minmax_element(IntPtr p_vec)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		size_t id_min, id_max;
		if (!TRTC_MinMax_Element(*vec, id_min, id_max))
			return nullptr;		
		return gcnew Tuple<int64_t, int64_t>{(int64_t)id_min, (int64_t)id_max};
	}

	Tuple<int64_t, int64_t>^ Native::minmax_element(IntPtr p_vec, IntPtr p_comp)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* comp = just_cast_it<Functor>(p_comp);
		size_t id_min, id_max;
		if (!TRTC_MinMax_Element(*vec, *comp, id_min, id_max))
			return nullptr;
		return gcnew Tuple<int64_t, int64_t>{(int64_t)id_min, (int64_t)id_max};
	}

	Object^ Native::inner_product(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_init)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		DeviceViewable* init = just_cast_it<DeviceViewable>(p_init);
		ViewBuf ret;
		if (!TRTC_Inner_Product(*vec1, *vec2, *init, ret))
			return nullptr;
		return s_box_basic_type(ret, init->name_view_cls());		
	}

	Object^ Native::inner_product(IntPtr p_vec1, IntPtr p_vec2, IntPtr p_init, IntPtr p_binary_op1, IntPtr p_binary_op2)
	{
		DVVectorLike* vec1 = just_cast_it<DVVectorLike>(p_vec1);
		DVVectorLike* vec2 = just_cast_it<DVVectorLike>(p_vec2);
		DeviceViewable* init = just_cast_it<DeviceViewable>(p_init);
		Functor* binary_op1 = just_cast_it<Functor>(p_binary_op1);
		Functor* binary_op2 = just_cast_it<Functor>(p_binary_op2);
		ViewBuf ret;
		if (!TRTC_Inner_Product(*vec1, *vec2, *init, ret, *binary_op1, *binary_op2))
			return nullptr;
		return s_box_basic_type(ret, init->name_view_cls());
	}

	Object^ Native::transform_reduce(IntPtr p_vec, IntPtr p_unary_op, IntPtr p_init, IntPtr p_binary_op)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* unary_op = just_cast_it<Functor>(p_unary_op);
		DeviceViewable* init = just_cast_it<DeviceViewable>(p_init);
		Functor* binary_op = just_cast_it<Functor>(p_binary_op);
		ViewBuf ret;
		if (!TRTC_Transform_Reduce(*vec, *unary_op, *init, *binary_op, ret))
			return nullptr;
		return s_box_basic_type(ret, init->name_view_cls());
	}

	Object^ Native::all_of(IntPtr p_vec, IntPtr p_pred)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		bool ret;
		if (!TRTC_All_Of(*vec, *pred, ret))
			return nullptr;
		return ret;
	}

	Object^ Native::any_of(IntPtr p_vec, IntPtr p_pred)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		bool ret;
		if (!TRTC_Any_Of(*vec, *pred, ret))
			return nullptr;
		return ret;
	}

	Object^ Native::none_of(IntPtr p_vec, IntPtr p_pred)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		bool ret;
		if (!TRTC_None_Of(*vec, *pred, ret))
			return nullptr;
		return ret;
	}

	Object^ Native::is_partitioned(IntPtr p_vec, IntPtr p_pred)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* pred = just_cast_it<Functor>(p_pred);
		bool ret;
		if (!TRTC_Is_Partitioned(*vec, *pred, ret))
			return nullptr;
		return ret;
	}

	Object^ Native::is_sorted(IntPtr p_vec)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		bool ret;
		if (!TRTC_Is_Sorted(*vec, ret))
			return nullptr;
		return ret;
	}

	Object^ Native::is_sorted(IntPtr p_vec, IntPtr p_comp)
	{
		DVVectorLike* vec = just_cast_it<DVVectorLike>(p_vec);
		Functor* comp = just_cast_it<Functor>(p_comp);
		bool ret;
		if (!TRTC_Is_Sorted(*vec, *comp, ret))
			return nullptr;
		return ret;
	}
}

