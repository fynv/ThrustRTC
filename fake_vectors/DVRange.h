#ifndef _DVRange_h
#define _DVRange_h

#include "DVVector.h"

class THRUST_RTC_API DVRange : public DVVectorLike
{
public:
	std::string cls_value() const { return m_cls_value; }
	ViewBuf view_value() const { return m_view_value; }

	DVRange(const DVVectorLike& vec_value, size_t begin = 0, size_t end = (size_t)(-1));
	virtual ViewBuf view() const;
	virtual bool is_readable() const { return m_readable; }
	virtual bool is_writable() const { return m_writable; }

private:
	bool m_readable;
	bool m_writable;
	std::string m_cls_value;
	ViewBuf m_view_value;
	size_t m_begin;
	size_t m_end;

	size_t m_offsets[4];

};


#endif

