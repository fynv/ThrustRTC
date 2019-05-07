#include <stdio.h>
#include <memory.h>
#include "DVTransform.h"

DVTransform::DVTransform(TRTCContext& ctx, const DVVectorLike& vec_in, const char* elem_cls, const Functor& op)
	: DVVectorLike(ctx, elem_cls, vec_in.size())
{
	m_name_view_cls = std::string("TransformView<") + elem_cls + "," + vec_in.name_view_cls() + "," + op.name_view_cls() + ">";
	m_view_in = vec_in.view();
	m_view_op = op.view();
}

std::string DVTransform::name_view_cls() const
{
	return m_name_view_cls;
}

ViewBuf DVTransform::view() const
{
	size_t total = m_view_in.size() + m_view_op.size();
	ViewBuf buf(total);
	memcpy(buf.data(), m_view_in.data(), m_view_in.size());
	memcpy(buf.data() + m_view_in.size(), m_view_op.data(), m_view_op.size());
	return buf;
}
