#include <stdio.h>
#include <memory.h>
#include "DVTransform.h"

DVTransform::DVTransform(TRTCContext& ctx, const DVVectorLike& vec_in, const char* elem_cls, const Functor& op)
	: m_ctx(ctx), DVVectorLike(ctx, elem_cls, vec_in.size())
{
	char buf[64];
	int id = ctx.next_identifier();
	sprintf(buf, "_TransformView_%d", id);
	m_name_view_cls = buf;

	std::string struct_def = "#pragma pack(1)\n";
	struct_def += std::string("struct ") + m_name_view_cls + "\n{\n"
		"    typedef " + elem_cls + " value_t;\n"
		"    " + vec_in.name_view_cls() + " _view_vec_in;\n";

	for (size_t i = 0; i < op.arg_map.size(); i++)
		struct_def += std::string("    ") + op.arg_map[i].arg->name_view_cls() + " " + op.arg_map[i].param_name + ";\n";

	struct_def += "\n    __device__ size_t size() const\n    {\n"
		"        return _view_vec_in.size();\n    }\n";

	struct_def += std::string("\n    __device__ value_t& operator [](size_t _idx)\n    {\n") +
		op.generate_code(elem_cls, { "_view_vec_in[_idx]" }) +
		"        return " + op.functor_ret + ";\n    }\n};";

	ctx.add_code_block(struct_def.c_str());

	m_view_in = vec_in.view();
	m_view_params.resize(op.arg_map.size());
	for (size_t i = 0; i < op.arg_map.size(); i++)
		m_view_params[i] = op.arg_map[i].arg->view();
}

std::string DVTransform::name_view_cls() const
{
	return m_name_view_cls;
}

ViewBuf DVTransform::view() const
{
	size_t total = m_view_in.size();
	for (size_t i = 0; i < m_view_params.size(); i++)
		total += m_view_params[i].size();
	ViewBuf buf(total);
	memcpy(buf.data(), m_view_in.data(), m_view_in.size());
	char* p = &buf[m_view_in.size()];
	for (size_t i = 0; i < m_view_params.size(); i++)
	{
		memcpy(p, m_view_params[i].data(), m_view_params[i].size());
		p += m_view_params[i].size();
	}
	return buf;
}
