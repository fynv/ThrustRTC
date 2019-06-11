#include <memory.h>
#include "DVReverse.h"

DVReverse::DVReverse(const DVVectorLike& vec_value)
	: DVVectorLike(vec_value.name_elem_cls().c_str(), vec_value.name_ref_type().c_str(), vec_value.size())
{
	m_readable = vec_value.is_readable();
	m_writable = vec_value.is_writable();
	m_cls_value = vec_value.name_view_cls();
	m_view_value = vec_value.view();
}


std::string DVReverse::name_view_cls() const
{
	return std::string("ReverseView<") + m_cls_value + ">";
}


ViewBuf DVReverse::view() const
{
	ViewBuf buf(m_view_value.size());
	memcpy(buf.data(), m_view_value.data(), m_view_value.size());
	return buf;
}
