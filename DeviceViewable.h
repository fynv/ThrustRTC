#ifndef _DeviceViewable_h
#define _DeviceViewable_h

#include <string>
#include <vector>

typedef std::vector<char> ViewBuf;

// root class of all device-viewable objects
class DeviceViewable
{
public:
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


#endif
