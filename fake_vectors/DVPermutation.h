#ifndef _DVPermutation_h
#define _DVPermutation_h

#include "DVVector.h"

class THRUST_RTC_API DVPermutation : public DVVectorLike
{
public:
	std::string cls_value() const { return m_cls_value; }
	ViewBuf view_value() const { return m_view_value; }
	std::string cls_index() const { return m_cls_index; }
	ViewBuf view_index() const { return m_view_index;  }

	DVPermutation(const DVVectorLike& vec_value, const DVVectorLike& vec_index );
	virtual std::string name_view_cls() const;
	virtual ViewBuf view() const;
	virtual bool is_readable() const { return m_readable; }
	virtual bool is_writable() const { return m_writable; }

private:
	bool m_readable;
	bool m_writable;
	std::string m_cls_value;
	ViewBuf m_view_value;
	std::string m_cls_index;
	ViewBuf m_view_index;

	size_t m_offsets[3];
};

#endif
