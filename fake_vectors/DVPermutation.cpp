#include <memory.h>
#include "DVPermutation.h"

DVPermutation::DVPermutation(TRTCContext& ctx, const DVVectorLike& vec_value, const DVVectorLike& vec_index)
	: DVVectorLike(ctx, vec_value.name_elem_cls().c_str(), vec_index.size())
{
	m_cls_value = vec_value.name_view_cls();
	m_view_value = vec_value.view();
	m_cls_index = vec_index.name_view_cls();
	m_view_index = vec_index.view();

	std::string name_struct = name_view_cls();
	ctx.query_struct(name_struct.c_str(), { "_view_vec_value", "_view_vec_index" }, m_offsets);
}

std::string DVPermutation::name_view_cls() const
{
	return std::string("PermutationView<") + m_cls_value + ", "+ m_cls_index+">";
}

ViewBuf DVPermutation::view() const
{
	ViewBuf buf(m_offsets[2]);
	memcpy(buf.data() + m_offsets[0], m_view_value.data(), m_view_value.size());
	memcpy(buf.data() + m_offsets[1], m_view_index.data(), m_view_index.size());
	return buf;
}

