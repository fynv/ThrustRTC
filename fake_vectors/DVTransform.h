#ifndef _DVTransform_h
#define _DVTransform_h

#include "DVVector.h"

template<class _T, class _T_VIN, class _T_OP>
struct TransformView
{
	typedef _T value_t;
	_T_VIN _view_vec_in;
	_T_OP _view_op;

#ifdef DEVICE_ONLY
	__device__ size_t size() const
	{
		return _view_vec_in.size();
	}

	__device__ value_t operator [](size_t idx)
	{
		return _view_op(_view_vec_in[idx]);
	}
#endif
};

#ifndef DEVICE_ONLY
#include "functor.h"

class THRUST_RTC_API DVTransform : public DVVectorLike
{
public:
	DVTransform(TRTCContext& ctx, const DVVectorLike& vec_in, const char* elem_cls, const Functor& op);

	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;

private:
	std::string m_name_view_cls;
	ViewBuf m_view_in;
	ViewBuf m_view_op;
	size_t m_offsets[3];
};

#endif 

#endif