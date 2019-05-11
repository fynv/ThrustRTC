#ifndef _DVDiscard_h
#define _DVDiscard_h

#include "DVVector.h"

template<class _T>
struct _Sink
{
#ifdef DEVICE_ONLY
	__device__ const _T& operator = (const _T& in)
	{
		return in;
	}
#endif
};

template<class _T>
struct DiscardView
{
	typedef _T value_t;
	typedef _Sink<_T>& ref_t;
	size_t _size;
	_Sink<_T> _sink;

#ifdef DEVICE_ONLY
	__device__ size_t size() const
	{
		return _size;
	}

	__device__ ref_t operator [](size_t)
	{
		return _sink;
	}
#endif
};

#ifndef DEVICE_ONLY

class THRUST_RTC_API DVDiscard : public DVVectorLike
{
public:
	DVDiscard(TRTCContext& ctx, const char* elem_cls, size_t size = (size_t)(-1));
	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;
	virtual bool is_readable() const { return false; }
	virtual bool is_writable() const { return true; }

};

#endif 

#endif
