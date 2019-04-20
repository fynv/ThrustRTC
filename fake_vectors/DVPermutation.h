#ifndef _DVPermutation_h
#define _DVPermutation_h

#include "DVVector.h"

template<class _TVVALUE, class _TVINDEX>
#pragma pack(1)
struct PermutationView
{
	typedef typename _TVVALUE::value_t value_t;
	_TVVALUE view_vec_value;
	_TVINDEX view_vec_index;

#ifdef DEVICE_ONLY
	__device__ value_t& operator [](size_t idx)
	{
		return view_vec_value[view_vec_index[idx]];
	}
#endif
};


#ifndef DEVICE_ONLY

class THRUST_RTC_API DVPermutation : public DVVectorLike
{
public:
	std::string cls_value() const { return m_cls_value; }
	ViewBuf view_value() const { return m_view_value; }
	std::string cls_index() const { return m_cls_index; }
	ViewBuf view_index() const { return m_view_index;  }

	DVPermutation(TRTCContext& ctx, const DVVectorLike& vec_value, const DVVectorLike& vec_index );
	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;

private:
	std::string m_cls_value;
	ViewBuf m_view_value;
	std::string m_cls_index;
	ViewBuf m_view_index;
};

#endif 

#endif
