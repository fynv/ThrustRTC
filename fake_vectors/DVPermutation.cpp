#include "DVPermutation.h"

DVPermutation::DVPermutation(TRTCContext& ctx, const DVVectorLike& vec_value, const DVVectorLike& vec_index)
	: DVVectorLike(ctx, vec_value.name_elem_cls().c_str(), vec_index.size())
{
	m_cls_value = vec_value.name_view_cls();
	m_view_value = vec_value.view();
	m_cls_index = vec_index.name_view_cls();
	m_view_index = vec_index.view();
}

std::string DVPermutation::name_view_cls() const
{
	return std::string("PermutationView<") + m_cls_value + ", "+ m_cls_index+">";
}

ViewBuf DVPermutation::view() const
{
	ViewBuf buf(m_view_value.size()+ m_view_index.size());
	memcpy(buf.data(), m_view_value.data(), m_view_value.size());
	memcpy(buf.data() + m_view_value.size(), m_view_index.data(), m_view_index.size());
	return buf;
}

