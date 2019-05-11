#include <stdio.h>
#include <memory.h>
#include "DVTransform.h"

DVTransform::DVTransform(TRTCContext& ctx, const DVVectorLike& vec_in, const char* elem_cls, const Functor& op)
	: DVVectorLike(ctx, elem_cls, elem_cls, vec_in.size())
{
	m_name_view_cls = std::string("TransformView<") + elem_cls + "," + vec_in.name_view_cls() + "," + op.name_view_cls() + ">";
	m_view_in = vec_in.view();
	m_view_op = op.view();
	ctx.query_struct(m_name_view_cls.c_str(), { "_view_vec_in", "_view_op" }, m_offsets);
}

std::string DVTransform::name_view_cls() const
{
	return m_name_view_cls;
}

ViewBuf DVTransform::view() const
{
	ViewBuf buf(m_offsets[2]);
	memcpy(buf.data() + m_offsets[0], m_view_in.data(), m_view_in.size());
	memcpy(buf.data() + m_offsets[1], m_view_op.data(), m_view_op.size());
	return buf;
}
