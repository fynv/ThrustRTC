#ifndef _DeviceViewable_h
#define _DeviceViewable_h

#include <string>
#include <vector>
#include <cstdint>
#include <memory.h>

typedef std::vector<char> ViewBuf;

// root class of all device-viewable objects
class DeviceViewable
{
public:
	DeviceViewable(){}
	virtual ~DeviceViewable(){}
	virtual ViewBuf view() const = 0;
	const std::string& name_view_cls() const { return m_name_view_cls; }

protected:
	std::string m_name_view_cls;
};


struct CapturedDeviceViewable
{
	const char* obj_name;
	const DeviceViewable* obj;
};


class SomeDeviceViewable : public DeviceViewable
{
public:
	SomeDeviceViewable(const char* name_view_cls, const void* data_view = "", size_t size_view = 1)
	{
		m_name_view_cls = name_view_cls;
		m_view_buf.resize(size_view);
		memcpy(m_view_buf.data(), data_view, size_view);
	}

	virtual ViewBuf view() const
	{
		return m_view_buf;
	}

private:
	ViewBuf m_view_buf;
};

#define DECLAR_DV_BASIC(clsname, type)\
class clsname : public SomeDeviceViewable\
{\
public:\
	clsname(type in) : SomeDeviceViewable(#type, &in, sizeof(type)) {}\
};

DECLAR_DV_BASIC(DVChar, char)
DECLAR_DV_BASIC(DVSChar, signed char)
DECLAR_DV_BASIC(DVUChar, unsigned char)
DECLAR_DV_BASIC(DVShort, short)
DECLAR_DV_BASIC(DVUShort, unsigned short)
DECLAR_DV_BASIC(DVInt, int)
DECLAR_DV_BASIC(DVUInt, unsigned int)
DECLAR_DV_BASIC(DVLong, long)
DECLAR_DV_BASIC(DVULong, unsigned long)
DECLAR_DV_BASIC(DVLongLong, long long)
DECLAR_DV_BASIC(DVULongLong, unsigned long long)
DECLAR_DV_BASIC(DVFloat, float)
DECLAR_DV_BASIC(DVDouble, double)
DECLAR_DV_BASIC(DVBool, bool)

DECLAR_DV_BASIC(DVInt8, int8_t)
DECLAR_DV_BASIC(DVUInt8, uint8_t)
DECLAR_DV_BASIC(DVInt16, int16_t)
DECLAR_DV_BASIC(DVUInt16, uint16_t)
DECLAR_DV_BASIC(DVInt32, int32_t)
DECLAR_DV_BASIC(DVUInt32, uint32_t)
DECLAR_DV_BASIC(DVInt64, int64_t)
DECLAR_DV_BASIC(DVUInt64, uint64_t)

DECLAR_DV_BASIC(DVSizeT, size_t)

inline DeviceViewable* dv_from_viewbuf(const ViewBuf& buf, const char* type)
{
	std::string s_type = type;
	if (s_type == "int8_t")
		return new DVInt8(*(int8_t*)buf.data());
	else if (s_type == "uint8_t")
		return new DVUInt8(*(uint8_t*)buf.data());
	else if (s_type == "int16_t")
		return new DVInt16(*(int16_t*)buf.data());
	else if (s_type == "uint16_t")
		return new DVUInt16(*(uint16_t*)buf.data());
	else if (s_type == "int32_t")
		return new DVInt32(*(int32_t*)buf.data());
	else if (s_type == "uint32_t")
		return new DVUInt32(*(uint32_t*)buf.data());
	else if (s_type == "int64_t")
		return new DVInt64(*(int64_t*)buf.data());
	else if (s_type == "uint64_t")
		return new DVUInt64(*(uint64_t*)buf.data());
	else if (s_type == "float")
		return new DVFloat(*(float*)buf.data());
	else if (s_type == "double")
		return new DVDouble(*(double*)buf.data());
	else if (s_type == "bool")
		return new DVBool(*(bool*)buf.data());
	else
		return new SomeDeviceViewable(type, buf.data(), buf.size());
}

#endif
