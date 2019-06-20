#include <memory.h>
#include "DVRange.h"

DVRange::DVRange(const DVVectorLike& vec_value, size_t begin, size_t end)
	: DVVectorLike(vec_value.name_elem_cls().c_str(), vec_value.name_ref_type().c_str(), (end == (size_t)(-1) ? vec_value.size() : end) - begin)
{
	if (end == (size_t)(-1)) end = vec_value.size();
	m_readable = vec_value.is_readable();
	m_writable = vec_value.is_writable();
	m_cls_value = vec_value.name_view_cls();
	m_view_value = vec_value.view();
	m_begin = begin;
	m_end = end;

	std::string name_struct = name_view_cls();
	TRTC_Query_Struct(name_struct.c_str(), { "_view_vec_value", "_begin", "_end" }, m_offsets);
}

std::string DVRange::name_view_cls() const
{
	return std::string("RangeView<") + m_cls_value + ">";
}

ViewBuf DVRange::view() const
{
	ViewBuf buf(m_offsets[3]);
	memcpy(buf.data() + m_offsets[0], m_view_value.data(), m_view_value.size());
	memcpy(buf.data() + m_offsets[1], &m_begin, sizeof(size_t));
	memcpy(buf.data() + m_offsets[2], &m_end, sizeof(size_t));
	return buf;
}


