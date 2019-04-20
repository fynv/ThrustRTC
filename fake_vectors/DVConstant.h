#ifndef _DVConstant_h
#define _DVConstant_h

#include "DVVector.h"

template<class _T>
#pragma pack(1)
struct ConstantView
{
	typedef _T value_t;
	value_t _value;
	size_t _size;

#ifdef DEVICE_ONLY
	__device__ size_t size() const
	{
		return _size;
	}

	__device__ const value_t& operator [](size_t)
	{
		return _value;
	}
#endif
};


#ifndef DEVICE_ONLY

class THRUST_RTC_API DVConstant : public DVVectorLike
{
public:
	ViewBuf value() const { return m_value; }
	DVConstant(TRTCContext& ctx, const DeviceViewable& dvobj, size_t size = (size_t)(-1));
	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;

private:
	ViewBuf m_value;
};

#endif 

#endif
