#ifndef _DVCounter_h
#define _DVCounter_h

#include "DVVector.h"

template<class _T>
struct CounterView
{
	typedef _T value_t;
	size_t _size;
	value_t _value_init;

#ifdef DEVICE_ONLY
	__device__ size_t size() const
	{
		return _size;
	}

	__device__ value_t operator [](size_t idx)
	{
		return _value_init+(value_t)idx;
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
	size_t m_offsets[3];
};

#endif 

#endif
