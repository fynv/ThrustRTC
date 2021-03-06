#include <memory.h>
#include "DVPermutation.h"

DVPermutation::DVPermutation(const DVVectorLike& vec_value, const DVVectorLike& vec_index)
	: DVVectorLike(vec_value.name_elem_cls().c_str(), vec_value.name_ref_type().c_str(), vec_index.size())
{
	m_readable = vec_value.is_readable();
	m_writable = vec_value.is_writable();
	m_cls_value = vec_value.name_view_cls();
	m_view_value = vec_value.view();
	m_cls_index = vec_index.name_view_cls();
	m_view_index = vec_index.view();

	m_name_view_cls = std::string("PermutationView<") + m_cls_value + ", " + m_cls_index + ">";
	TRTC_Query_Struct(m_name_view_cls.c_str(), { "_view_vec_value", "_view_vec_index" }, m_offsets);

}

ViewBuf DVPermutation::view() const
{
	ViewBuf buf(m_offsets[2]);
	memcpy(buf.data() + m_offsets[0], m_view_value.data(), m_view_value.size());
	memcpy(buf.data() + m_offsets[1], m_view_index.data(), m_view_index.size());
	return buf;
}

