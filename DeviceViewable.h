#ifndef _DeviceViewable_h
#define _DeviceViewable_h

#include <string>
#include <vector>
#include <cstdint>

typedef std::vector<char> ViewBuf;

// root class of all device-viewable objects
class DeviceViewable
{
public:
	DeviceViewable(){}
	virtual ~DeviceViewable(){}
	virtual std::string name_view_cls() const = 0;
	virtual ViewBuf view() const = 0;
};

#define DECLAR_DV_BASIC(clsname, type)\
class clsname : public DeviceViewable\
{\
public:\
	clsname(type in) : m_value(in) {}\
	virtual std::string name_view_cls() const\
	{\
		return #type;\
	}\
	virtual ViewBuf view() const\
	{\
		ViewBuf buf(sizeof(type));\
		*(type*)buf.data() = m_value;\
		return buf;\
	}\
private:\
	type m_value;\
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

#endif
