#ifndef _DVCounter_h
#define _DVCounter_h

#include "DVVector.h"

template<class _T>
#pragma pack(1)
struct CounterView
{
	typedef _T value_t;
	value_t value_init;
	size_t size;

#ifdef DEVICE_ONLY
	__device__ value_t operator [](size_t idx)
	{
		return value_init+(value_t)idx;
	}
#endif
};


#ifndef DEVICE_ONLY

class THRUST_RTC_API DVCounter : public DVVectorLike
{
public:
	ViewBuf value_init() const { return m_value_init; }
	DVCounter(TRTCContext& ctx, const DeviceViewable& dvobj_init, size_t size = (size_t)(-1));
	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;

private:
	ViewBuf m_value_init;
};

#endif 

#endif
