#ifndef _DVTransform_h
#define _DVTransform_h

#include "DVVector.h"
#include "functor.h"

class THRUST_RTC_API DVTransform : public DVVectorLike
{
public:
	DVTransform(const DVVectorLike& vec_in, const char* elem_cls, const Functor& op);

	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;

private:
	std::string m_name_view_cls;
	ViewBuf m_view_in;
	ViewBuf m_view_op;
	size_t m_offsets[3];
};

#endif
