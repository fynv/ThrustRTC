#include "stdafx.h"
#include "ThrustRTCLR.h"
#include "DeviceViewable.h"

namespace ThrustRTCLR
{
	template<typename T>
	inline T* just_cast_it(IntPtr p)
	{
		return (T*)(void*)p;
	}

	String^ Native::dv_name_view_cls(IntPtr p_dv)
	{
		DeviceViewable* dv = just_cast_it<DeviceViewable>(p_dv);
		return gcnew String(dv->name_view_cls().c_str());
	}

	void Native::dv_destroy(IntPtr p_dv)
	{
		DeviceViewable* dv = just_cast_it<DeviceViewable>(p_dv);
		delete dv;
	}

	IntPtr Native::dvint8_create(int8_t v)
	{
		return (IntPtr)(new DVInt8(v));
	}

	int8_t Native::dvint8_value(IntPtr p)
	{
		return *(int8_t*)just_cast_it<DVInt8>(p)->view().data();
	}

	IntPtr Native::dvuint8_create(uint8_t v)
	{
		return (IntPtr)(new DVUInt8(v));
	}

	uint8_t Native::dvuint8_value(IntPtr p)
	{
		return *(uint8_t*)just_cast_it<DVUInt8>(p)->view().data();
	}

	IntPtr Native::dvint16_create(int16_t v)
	{
		return (IntPtr)(new DVInt16(v));
	}

	int16_t Native::dvint16_value(IntPtr p)
	{
		return *(int16_t*)just_cast_it<DVInt16>(p)->view().data();
	}

	IntPtr Native::dvuint16_create(uint16_t v)
	{
		return (IntPtr)(new DVUInt16(v));
	}

	uint16_t Native::dvuint16_value(IntPtr p)
	{
		return *(uint16_t*)just_cast_it<DVUInt16>(p)->view().data();
	}

	IntPtr Native::dvint32_create(int32_t v)
	{
		return (IntPtr)(new DVInt32(v));
	}

	int32_t Native::dvint32_value(IntPtr p)
	{
		return *(int32_t*)just_cast_it<DVInt32>(p)->view().data();
	}

	IntPtr Native::dvuint32_create(uint32_t v)
	{
		return (IntPtr)(new DVUInt32(v));
	}

	uint32_t Native::dvuint32_value(IntPtr p)
	{
		return *(uint32_t*)just_cast_it<DVUInt32>(p)->view().data();
	}

	IntPtr Native::dvint64_create(int64_t v)
	{
		return (IntPtr)(new DVInt64(v));
	}

	int64_t Native::dvint64_value(IntPtr p)
	{
		return *(int64_t*)just_cast_it<DVInt64>(p)->view().data();
	}

	IntPtr Native::dvuint64_create(uint64_t v)
	{
		return (IntPtr)(new DVUInt64(v));
	}

	uint64_t Native::dvuint64_value(IntPtr p)
	{
		return *(uint64_t*)just_cast_it<DVUInt64>(p)->view().data();
	}

	IntPtr Native::dvfloat_create(float v)
	{
		return (IntPtr)(new DVFloat(v));
	}

	float Native::dvfloat_value(IntPtr p)
	{
		return *(float*)just_cast_it<DVFloat>(p)->view().data();
	}

	IntPtr Native::dvdouble_create(double v)
	{
		return (IntPtr)(new DVDouble(v));
	}

	double Native::dvdouble_value(IntPtr p)
	{
		return *(double*)just_cast_it<DVDouble>(p)->view().data();
	}

	IntPtr Native::dvbool_create(bool v)
	{
		return (IntPtr)(new DVBool(v));
	}

	bool Native::dvbool_value(IntPtr p)
	{
		return *(bool*)just_cast_it<DVBool>(p)->view().data();
	}

}
