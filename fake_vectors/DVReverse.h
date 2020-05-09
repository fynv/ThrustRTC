#ifndef _DVReverse_h
#define _DVReverse_h

#include "DVVector.h"

class THRUST_RTC_API DVReverse : public DVVectorLike
{
public:
	std::string cls_value() const { return m_cls_value; }
	ViewBuf view_value() const { return m_view_value; }

	DVReverse(const DVVectorLike& vec_value);
	virtual ViewBuf view() const;
	virtual bool is_readable() const { return m_readable; }
	virtual bool is_writable() const { return m_writable; }

private:
	bool m_readable;
	bool m_writable;
	std::string m_cls_value;
	ViewBuf m_view_value;
};


#endif
